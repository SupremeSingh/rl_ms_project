import importlib.util
from pathlib import Path

import pytest
import torch


def audit_module():
    path=Path(__file__).resolve().parents[1]/'tools/encoder/audit_study.py'
    spec=importlib.util.spec_from_file_location('encoder_audit',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_audit_metrics_keep_questions_equal_and_report_unbounded_predictions():
    m=audit_module()
    rows=[dict(question='a',positions=[0,2],y=torch.tensor([0.,0.])),
          dict(question='b',positions=[0],y=torch.tensor([1.]))]
    result=m.metrics(rows,[torch.tensor([0.,2.]),torch.tensor([0.])])
    assert result['question_mean_brier']==1.5
    assert result['prefix_mean_brier']==pytest.approx(5/3)
    assert result['initial_state_brier']==.5
    assert result['outside_unit_interval']==pytest.approx(1/3)


def test_completed_audit_restores_models_and_never_changes_source(tmp_path,monkeypatch):
    a=audit_module();m=a.study
    rows=[dict(x=torch.randn(3,10),y=torch.full((3,),float(i)),positions=[0,2,4],
               split=s,question=f'{s}-{i}') for s in ('train','val','test') for i in (0,1)]
    monkeypatch.setattr(m,'feature_rows',lambda *args:rows)
    recipe=dict(seeds=[42],compressor_epochs=1,model_identity={'test':'identity'},scope='Synthetic test')
    monkeypatch.setattr(m,'validate',lambda *args:recipe)
    m.write_json(tmp_path/'manifest.json',recipe)
    manifest_hash=m.sha256(tmp_path/'manifest.json')
    for arm in m.ARMS:
        folder=tmp_path/arm;folder.mkdir()
        m.write_json(folder/'features.json',dict(manifest_sha256=manifest_hash,
            model_identity=recipe['model_identity'],seconds=1,bidirectional_sensitivity=0 if arm=='causal' else 1))
    m.fit(tmp_path,recipe);m.evaluate(tmp_path,recipe)
    (tmp_path/'adaptation.txt').write_text('line\n')
    m.write_json(tmp_path/'adaptation.json',dict(model_identity=recipe['model_identity'],
        training_windows=1,seconds={},text_sha256=m.sha256(tmp_path/'adaptation.txt')))
    for stage in ('mntp','simcse'):m.write_json(tmp_path/f'{stage}.json',{})
    before={p.relative_to(tmp_path):m.sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    report=a.audit(tmp_path)
    assert report['checks_passed']
    assert report['arms']['adapted']['compressor_optimizer_steps']==1
    assert report['arms']['causal']['seeds']['42']['restored_validation_matches']
    after={p.relative_to(tmp_path):m.sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after
