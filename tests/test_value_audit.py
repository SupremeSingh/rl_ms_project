"""CPU acceptance of policy identity, causal alignment, MC caching and Slurm gates."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from math_rl import value_audit as audit
from math_rl import value_audit_gpu as gpu
from math_rl.provenance import sha256, write_json
from math_rl.value_features import normalize_readout

ROOT = Path(__file__).resolve().parents[1]


def controller():
    spec = importlib.util.spec_from_file_location('submit_value_audit', ROOT/'scripts/submit_value_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return dict(actor_seed=1001, steps=120, ppo_profile='higher_lr', arms=audit.ARMS,
                budgets=audit.BUDGETS, filter_sha256='fixed', layers=[27,26,24,20,12], rms_eps=1e-6,
                reference_seed=48371, grpo_group_size=4, response_cap=2048, reference_batch=2,
                sampling=dict(temperature=1., top_p=1., top_k=-1), max_model_len=2560)


def test_normalization_is_explicit_and_does_not_mutate_old_coordinates():
    x = np.array([[3.,4.],[-2.,1.]], dtype=np.float32)
    x.flags.writeable = False
    assert normalize_readout(x, 'none') is x
    expected = x/np.sqrt((x*x).mean(-1,keepdims=True)+1e-6)
    np.testing.assert_allclose(normalize_readout(x, 'rms'), expected)
    np.testing.assert_array_equal(x, [[3.,4.],[-2.,1.]])
    with pytest.raises(ValueError):normalize_readout(x, 'layernorm')


def test_matched_commands_export_real_checkpoints_and_train_true_rms_heads(tmp_path):
    m = manifest()
    for arm, setting in audit.ARMS.items():
        cmd, case = audit.command(tmp_path, m, arm)
        assert 'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]' in cmd
        assert f'actor_rollout_ref.model.critic_feature_normalization={setting["normalization"]}' in cmd
        assert f'experiment.chunking={setting["mode"]}' in cmd
        assert 'experiment.critic_folds=1' in cmd and 'experiment.cumulative_epsilon=0.01' in cmd
        assert 'experiment.evaluation_split=val' in cmd
        assert not any('data/test.parquet' in value for value in cmd)
        assert ('critic.checkpoint.save_contents=[model,hf_model]' in cmd) == (arm == 'ppo')
        assert case == tmp_path/f'{arm}-seed1001'
    assert audit.checkpoint_paths(tmp_path, m, 'slice-token', 'smoke')[1] == (
        tmp_path/'smoke-slice-token-seed19/global_step_2/actor/huggingface')


def test_boundary_selection_and_question_mixture_are_outcome_independent(tmp_path):
    write_json(tmp_path/'data/questions.json', dict(questions=[dict(id=str(i),split='test',
        question=f'Question {i}',ground_truth='1') for i in range(50)]))
    m=manifest()
    questions=audit.audit_questions(tmp_path,m,'pilot')
    assert len(questions)==10 and len({q['id'] for q in questions})==10
    assert [q['bank_arm'] for q in questions] == (list(audit.ARMS)*2)[:10]
    assert questions==audit.audit_questions(tmp_path,m,'pilot')
    assert audit.boundary_positions((0,10,20,30,40,50),4)==[0,20,30,50]
    assert audit.boundary_positions((0,),4)==[0]
    with pytest.raises(ValueError):audit.boundary_positions((2,3),4)


def test_packer_preserves_original_ids_and_remaining_budget(tmp_path,monkeypatch):
    m=manifest();phase='smoke';folder=tmp_path/phase;folder.mkdir()
    write_json(tmp_path/'manifest.json',{})
    q=dict(id='q',ground_truth='1',bank_arm='slice-token')
    monkeypatch.setattr(audit,'audit_questions',lambda *a:[q])
    for arm in audit.ARMS:
        rows=[dict(q,prompt_ids=[1,2],answer_ids=[999,77,88,2],positions=[0,2])] if arm=='slice-token' else []
        write_json(folder/f'bank-{arm}.json',dict(manifest_sha256=sha256(tmp_path/'manifest.json'),
            tokenizer_sha256='fixed',questions=rows))
    audit.pack_bank(tmp_path,m,phase)
    states=audit.read(folder/'bank.json')['states']
    assert states[1]['prefix_ids']==[999,77] and states[1]['remaining_tokens']==2046
    assert states[0]['prefix_ids']==[]
    bad=audit.read(folder/'bank-slice-token.json');bad['questions'][0]['positions']=[0,0]
    write_json(folder/'bank-slice-token.json',bad)
    with pytest.raises(ValueError,match='boundary'):audit.pack_bank(tmp_path,m,phase)


def test_scalar_critic_hf_roundtrip_and_causal_prefix_alignment(tmp_path):
    from transformers import Qwen2Config, Qwen2ForTokenClassification, AutoModelForTokenClassification
    torch.manual_seed(13)
    config=Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, num_labels=1,
                      classifier_dropout=0., attention_dropout=0.)
    model=Qwen2ForTokenClassification(config).eval()
    from math_rl.ppo_baseline import prepare_value_export
    model.config.architectures=['Qwen2ForCausalLM']  # The inherited base-model config.
    prepare_value_export(model)
    assert model.config.architectures==['Qwen2ForTokenClassification']
    model.save_pretrained(tmp_path)
    restored,info=AutoModelForTokenClassification.from_pretrained(tmp_path,output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    restored.eval()
    actual=gpu.last_prefix_value(restored,[1,2],[3,4])
    with torch.no_grad():
        full=restored(input_ids=torch.tensor([[1,2,3,4,5,6]])).logits[0,3,0].item()
    assert actual==pytest.approx(full,abs=1e-6)
    assert gpu.last_prefix_value(restored,[1,2],[])==pytest.approx(
        model(input_ids=torch.tensor([[1,2]])).logits[0,-1,0].item(),abs=1e-6)


def test_generator_keeps_original_ids_and_respects_remaining_tokens(monkeypatch):
    requests=[]
    class Params:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=Params))
    generator=object.__new__(gpu.Generator);generator.m=manifest()
    generator.tokenizer=SimpleNamespace(eos_token_id=2)
    def generate(prompts,params,**kwargs):
        requests.append((prompts,params))
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[17],stop_reason=None,finish_reason='stop')])]
    generator.model=SimpleNamespace(generate=generate)
    rows=generator.generate([1,999,18],1,5,131)
    assert rows[0]['token_ids']==[17,2]
    assert requests[0][0]==[dict(prompt_token_ids=[1,999,18])]
    assert requests[0][1].max_tokens==5 and requests[0][1].seed==131
    with pytest.raises(ValueError):generator.generate([1],1,0,131)
    with pytest.raises(ValueError):generator.generate([1]*600,1,2048,131)


def test_grpo_groups_are_independent_and_reference_caches_resume(tmp_path,monkeypatch):
    m=manifest();folder=tmp_path/'smoke';folder.mkdir()
    write_json(tmp_path/'manifest.json',{})
    states=[dict(state_id=f'q/{i}',question_id='q',ground_truth='1',prompt_ids=[1,2],
                 prefix_ids=[99]*i,remaining_tokens=2048-i) for i in range(2)]
    write_json(folder/'bank.json',dict(tokenizer_sha256='tokens',states=states,
                                     questions=[dict(id='q',ground_truth='1')]))
    calls=[]
    class Generator:
        tokenizer=SimpleNamespace()
        def __init__(self,*a):pass
        def generate(self,prompt,count,remaining,seed):
            calls.append(dict(prompt=prompt,count=count,remaining=remaining,seed=seed))
            return [dict(token_ids=[11,2],score=i%2,capped=False,
                         verifier_status='correct' if i%2 else 'incorrect') for i in range(count)]
    monkeypatch.setattr(gpu,'Generator',Generator)
    monkeypatch.setattr(gpu,'checkpoint_paths',lambda *a:(None,Path('actor'),None,None))
    monkeypatch.setattr(gpu,'tokenizer_identity',lambda *a:'tokens')
    monkeypatch.setattr(gpu,'grade',lambda generator,gold,generated,prefix:generated)
    gpu.monte_carlo(tmp_path,m,'smoke','grpo')
    assert calls[0]['prompt']==[1,2] and calls[0]['count']==4
    assert calls[-1]['prompt']==[1,2,99] and calls[-1]['remaining']==2047
    assert len({c['seed'] for c in calls})==len(calls)
    predictions=audit.read(folder/'predictions-grpo.json')['states']
    assert [v['prediction'] for v in predictions]==[.5,.5]
    assert all(v['samples']==4 for v in audit.read(folder/'reference-grpo-0.json')['states'])
    n=len(calls)
    (folder/'reference-grpo-0.json').unlink();(folder/'predictions-grpo.json').unlink()
    gpu.monte_carlo(tmp_path,m,'smoke','grpo')
    assert len(calls)==n  # Samples, including independent GRPO groups, are reused exactly.


def test_question_weighted_metrics_keep_out_of_range_predictions():
    rows=[dict(question_id='a',prediction=1.5,samples=256,successes=256),
          *[dict(question_id='b',prediction=0.,samples=256,successes=0) for _ in range(3)]]
    result=audit.summarize(rows,draws=100)
    assert result['mae']==.25 and result['outside_unit_interval']==.5
    assert result['rmse']==pytest.approx(np.sqrt(.125))
    assert result['reference_worst_case_se']==.03125
    assert len(result['question_mc_sensitivity_mae_95'])==2
    assert audit.wilson(0,256)[1]>0 and audit.wilson(256,256)[0]<1
    with pytest.raises(ValueError):audit.summarize([])


def test_report_renders_all_panels_and_locks_reference_artifacts(tmp_path):
    pytest.importorskip('matplotlib')
    m=manifest();m['scope']='Synthetic CPU acceptance fixture; no research result.'
    folder=tmp_path/'smoke';folder.mkdir();write_json(tmp_path/'manifest.json',{})
    rows=[dict(state_id=f'q/{i}',question_id='q',progress=i*.5) for i in range(2)]
    write_json(folder/'bank.json',dict(states=rows))
    for arm in audit.ARMS:
        header=dict(manifest_sha256=sha256(tmp_path/'manifest.json'),bank_sha256=sha256(folder/'bank.json'),seconds=.1)
        write_json(folder/f'predictions-{arm}.json',dict(header,states=[dict(state_id=r['state_id'],prediction=1.4 if i else 0.) for i,r in enumerate(rows)]))
        write_json(folder/f'reference-{arm}-0.json',dict(header,states=[dict(state_id=r['state_id'],samples=4,successes=2) for r in rows],
            label_statuses={'correct':4,'incorrect':4},response_tokens=16,capped_continuations=0,cached_generation_and_verification_seconds=.1))
    audit.report(tmp_path,m,'smoke')
    audit.require_gate(tmp_path,'smoke/verification.json')
    assert (folder/'value-scatter.pdf').read_bytes().startswith(b'%PDF')
    summary=audit.read(folder/'summary.json')
    assert len(summary['metrics'])==6 and summary['panels']['slice-token'][1]['prediction']==1.4
    assert summary['metrics']['slice-token']['outside_unit_interval']==.5
    write_json(folder/'predictions-grpo.json',dict(changed=True))
    with pytest.raises(ValueError,match='artifact changed'):audit.require_gate(tmp_path,'smoke/verification.json')


def write_gate(out,phase):
    out.mkdir(exist_ok=True)
    write_json(out/'manifest.json',{})
    write_json(out/phase/'bank.json',{})
    write_json(out/phase/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),
        bank_sha256=sha256(out/phase/'bank.json')))


def test_smoke_submission_dependencies_and_resource_sizes(tmp_path,monkeypatch):
    m=controller();source=tmp_path/'source';source.mkdir();write_json(source/'manifest.json',{})
    calls=[]
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(200+len(calls)),stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    m.submit(tmp_path/'out','smoke',source,concurrency=3)
    assert len(calls)==7
    assert '--gres=gpu:a5000:2' in calls[1] and '--array=0-5%3' in calls[1]
    assert '--gres=gpu:a5000:1' in calls[3] and '--gres=gpu:a5000:1' in calls[5]
    for index,row in enumerate(calls[1:],1):assert f'--dependency=afterok:{200+index}' in row


def test_full_reuses_actors_and_shards_mc_after_predictions(tmp_path,monkeypatch):
    m=controller();out=tmp_path/'out';write_gate(out,'pilot');calls=[]
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    m.submit(out,'full',concurrency=4)
    assert len(calls)==5 and '--array=0-23%4' in calls[3]
    assert '--dependency=afterok:3' in calls[3]
    assert not any('--gres=gpu:a5000:2' in row for row in calls)


def test_ambiguous_sbatch_stops_without_duplicate_submission(tmp_path,monkeypatch):
    m=controller();source=tmp_path/'source';source.mkdir();write_json(source/'manifest.json',{})
    calls=[]
    def run(args,**kw):
        calls.append(args)
        raise subprocess.TimeoutExpired(args,90)
    monkeypatch.setattr(m.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='check Slurm'):m.submit(tmp_path/'out','smoke',source)
    with pytest.raises(FileExistsError):m.submit(tmp_path/'out','smoke',source)
    assert len(calls)==1


def test_full_requires_passed_pilot_with_unchanged_bank(tmp_path):
    m=controller();out=tmp_path/'out';write_gate(out,'pilot')
    write_json(out/'pilot/bank.json',dict(changed=True))
    with pytest.raises(ValueError,match='Matching passed pilot'):m.submit(out,'full')
    assert not (tmp_path/'out-submission/.submission-full').exists()
