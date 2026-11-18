import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
from math_rl import test_followup as study
from math_rl import cost_benchmark as cost

ROOT=Path(__file__).resolve().parents[1]
def controller():
    spec=importlib.util.spec_from_file_location('followup_submit',ROOT/'scripts/submit_test_followup.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_accuracy_intervals_pair_question_and_actor_seed():
    x=np.zeros((3,2000)); y=np.ones((3,2000))
    r=study.paired_accuracy(x,y,draws=100)
    assert r['mean_pp']==-100 and r['adjusted_interval_pp']==[-100.,-100.]
    assert study.paired_accuracy(x,x,draws=100)['adjusted_interval_pp']==[0.,0.]

def test_submission_has_all_18_actors_and_independent_base(tmp_path,monkeypatch):
    c=controller();calls=[]
    monkeypatch.setattr(c,'check_gate',lambda *a:None)
    monkeypatch.setattr(c.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')))
    c.submit(tmp_path/'new','main')
    assert '--array=0-18%6' in calls[0]
    assert '--dependency=afterok:1' in calls[1]
    assert all('--gres=gpu:a5000:1' in row for row in calls)
    assert all('--exclude=compsci-cluster-fitz-15,compsci-cluster-fitz-21' in row for row in calls)

def test_cost_comparisons_share_one_two_gpu_allocation(tmp_path,monkeypatch):
    c=controller();calls=[]
    monkeypatch.setattr(c.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')))
    c.submit(tmp_path/'new','benchmark-smoke',source=tmp_path/'source')
    assert len(calls)==2 and '--gres=gpu:a5000:2' in calls[1]
    assert not any(a.startswith('--array') for a in calls[1])

def test_ambiguous_submission_cannot_be_retried(tmp_path,monkeypatch):
    c=controller()
    def fail(*a,**kw):raise subprocess.TimeoutExpired('sbatch',90)
    monkeypatch.setattr(c.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='uncertain'):c.submit(tmp_path/'new','smoke',source=tmp_path/'source')
    with pytest.raises(FileExistsError):c.submit(tmp_path/'new','smoke',source=tmp_path/'source')

def test_fixed_workload_command_preserves_update_algorithm_and_separates_outputs(tmp_path):
    m=dict(source=str(tmp_path/'source'),feature_manifest=dict(filter_sha256='fixed',layers=[27,26,24,20,12],rms_eps=1e-6),warmup_updates=2,measured_updates=3,
           statistics={a:dict(path=f'/saved/{a}.pt') for a in study.ARMS[:4]})
    for a in study.ARMS:
        cmd,env,case=cost.trial_command(tmp_path,m,a,'accumulated',1)
        assert 'math_rl.benchmark_entry' in cmd and 'trainer.save_freq=-1' in cmd
        assert 'trainer.total_training_steps=5' in cmd and 'algorithm.lam=0.95' in cmd
        assert 'actor_rollout_ref.rollout.calculate_log_probs=false' in cmd
        assert case.parent==tmp_path/'trials'
        assert env['MATH_RL_BENCH_WORKLOAD']==str(tmp_path/'workload.json')
        assert env['MATH_RL_BENCH_TIMINGS']==str(case/'benchmark-steps.jsonl')
        assert ('MATH_RL_BENCH_STATISTICS' in env)==(a in study.ARMS[:4])
        if a=='ppo':assert 'critic.optim.lr=5e-05' in cmd and 'critic.ppo_epochs=2' in cmd
        assert not any('fixed_policy' in s for s in cmd)

def test_full_benchmark_requires_smoke_before_creating_output(tmp_path):
    # Guard must run before importing optional GPU/data libraries.
    # prepare currently imports datasets; this fixture checks CLI-level submission guard.
    c=controller()
    with pytest.raises(ValueError,match='smoke'):c.submit(tmp_path/'new','benchmark',source=tmp_path/'source')

def test_replay_keeps_original_action_ids_and_exact_group_size():
    # Exercise the pure adapter without importing the GPU-only VERL runtime.
    import ast
    tree=ast.parse((ROOT/'src/math_rl/benchmark_runtime.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay_outputs')
    namespace={'SimpleNamespace':SimpleNamespace}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'replay','exec'),namespace)
    replay=namespace['replay_outputs'];answers=[[900000,2],[8,2],[7,2],[9,2]]
    output=replay([dict(prompt_token_ids=[1,3])],4,{'1,3':answers})
    assert output[0].outputs[0].token_ids==[900000,2]
    output[0].outputs[0].token_ids.append(3)
    assert answers[0]==[900000,2]
    with pytest.raises(ValueError):replay([dict(prompt_token_ids=[1,3])],1,{'1,3':answers})

def test_imported_benchmark_entry_resolves_config_directory(tmp_path):
    import ast
    pytest.importorskip('hydra')
    config=tmp_path/'configs';config.mkdir();(config/'online.yaml').write_text('value: 42\n')
    tree=ast.parse((ROOT/'src/math_rl/benchmark_entry.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # Execute the actual decorator/function with CPU stubs for the Ray driver.
    code=('import hydra\nfrom pathlib import Path\nfrom types import SimpleNamespace\n'
          f'ROOT=Path({str(tmp_path)!r})\n'
          'def original(config):\n    print("CONFIG_VALUE="+str(config.value))\n'
          'module=SimpleNamespace(main=SimpleNamespace(__wrapped__=original))\n'
          +ast.unparse(main)+'\nmain()\n')
    script=tmp_path/'entry.py';script.write_text(code)
    result=subprocess.run([__import__('sys').executable,str(script)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'CONFIG_VALUE=42' in result.stdout

def test_ray_temp_directory_ignores_long_slurm_tmpdir(tmp_path,monkeypatch):
    import ast,tempfile
    monkeypatch.setenv('TMPDIR',str(tmp_path/('long-slurm-prefix-'*5)))
    tree=ast.parse((ROOT/'src/math_rl/cost_benchmark.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='TemporaryDirectory')
    kwargs={k.arg:ast.literal_eval(k.value) for k in call.keywords}
    with tempfile.TemporaryDirectory(**kwargs) as folder:
        suffix='/ray/session_2026-11-17_21-31-26_655458_1182212/sockets/plasma_store'
        assert len((folder+suffix).encode())<=107
        assert Path(folder).parent==Path('/tmp')

def test_reuse_workload_checks_source_hashes_without_reusing_failed_trials(tmp_path):
    from math_rl.provenance import sha256,write_json
    old=tmp_path/'old';old.mkdir();(old/'data').mkdir()
    (old/'data/train.parquet').write_bytes(b'fixed data')
    answers={str(i):[[1,2],[3,2],[4,2],[5,2]] for i in range(16)}
    write_json(old/'workload.json',dict(answers=answers,rewards={k:[0,1,0,1] for k in answers}))
    source=tmp_path/'source';features={'filter_sha256':'fixed'}
    write_json(old/'manifest.json',dict(protocol='fixed-workload-cost-v1',source=str(source),feature_manifest=features,
        workload_sha256=sha256(old/'workload.json'),data_hashes={'data/train.parquet':sha256(old/'data/train.parquet')},runtime={'old':'kept for audit'}))
    assert cost.reusable_workload(old,source,features)['runtime']=={'old':'kept for audit'}
    with pytest.raises(ValueError,match='differ'):cost.reusable_workload(old,source,{'filter_sha256':'other'})
    (old/'workload.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):cost.reusable_workload(old,source,features)

def test_structured_metrics_survive_missing_console_and_reject_partial_records(tmp_path):
    from math_rl.benchmark_metrics import append_metrics,read_steps
    path=tmp_path/'trial/benchmark-steps.jsonl'
    append_metrics(path,0,{'val/accuracy':.5},'workload')
    assert not path.exists()
    for step in [1,2,3]:
        append_metrics(path,step,{'timing_s/step':12.3456789,'timing_s/update_actor':8.,'perf/max_memory_allocated_gb':4.},'workload')
    records=read_steps(path,3,'workload')
    assert records[3]['timing_s/step']==12.3456789
    with pytest.raises(ValueError,match='identity'):read_steps(path,3,'other')
    with pytest.raises(ValueError,match='Missing'):read_steps(path,4,'workload')
    append_metrics(path,3,{'timing_s/step':10},'workload')
    with pytest.raises(ValueError,match='duplicated'):read_steps(path,3,'workload')
