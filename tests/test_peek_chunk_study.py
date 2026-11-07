import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def module():
    spec=importlib.util.spec_from_file_location('peek',ROOT/'tools/diagnostics/peek_chunk_study.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Characters:
    is_fast=True
    eos_token_id=0
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    def __call__(self,text,**kw):
        return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
    def encode(self,text,**kw):return list(map(ord,text))


def test_progress_reads_partial_runs_and_never_reports_training_as_test(tmp_path,capsys):
    m=module();case=tmp_path/'slice-chunk-seed19';case.mkdir()
    (case/'chunk-metrics.jsonl').write_text(json.dumps(dict(step=1,segmentation_fallback_answers=2))+'\n'+ '{"step":')
    m.progress(tmp_path)
    text=capsys.readouterr().out
    assert 'slice-chunk-seed19' in text and 'not finished' in text and 'not recorded in this older study' in text
    (case/'result.json').write_text(json.dumps(dict(updates=2,final_validation=dict(accuracy=.75))))
    m.progress(tmp_path)
    assert '75.00% (validation)' in capsys.readouterr().out


def test_latest_training_samples_round_robin_and_no_test_access(tmp_path):
    m=module()
    for case in ('a','b'):
        folder=tmp_path/case/'rollouts';folder.mkdir(parents=True)
        (folder/'2.jsonl').write_text(json.dumps(dict(output='old'))+'\n')
        (folder/'10.jsonl').write_text(''.join(json.dumps(dict(output=case+str(i)))+'\n' for i in range(3)))
        (folder.parent/'final-test.jsonl').write_text('not valid json: must never be read')
    rows=list(m.training_samples(tmp_path,3))
    assert [r['output'] for _,r in rows]==['a0','b0','a1']
    assert all(p.name=='10.jsonl' for p,_ in rows)


def test_chunk_report_distinguishes_estimated_tokens_and_shows_failure_example(tmp_path,capsys):
    m=module();folder=tmp_path/'run'/'rollouts';folder.mkdir(parents=True)
    rows=[dict(output='First compute the expression carefully. Then simplify the final result.'),
          dict(token_ids=list(map(ord,'$a$'*50))+[0])]
    (folder/'1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    m.inspect([tmp_path],Characters(),10,3,64)
    output=capsys.readouterr().out
    assert 'Observed segmentation failures: 1/2' in output
    assert 're-tokenized text (approximate)' in output
    assert 'EXAMPLE fallback' in output and '[64:128] (64 tokens)' in output
    assert 'Successful heuristic chunk tokens: n=' in output


def test_inspection_matches_current_runtime_for_normal_and_fallback():
    from slice_critic.chunks import token_chunks
    m=module()
    for text in ('First simplify carefully. Then calculate the final result.', '$a$'*50):
        for size in (16,64):
            tokens=list(map(ord,text))+[0]
            current=token_chunks(Characters(),tokens,fallback_tokens=size)
            peek=m.inspection_plan(Characters(),tokens,size)
            assert peek.starts==list(current.starts) and peek.ends==list(current.ends)
            assert peek.fallback_reason==current.fallback_reason
