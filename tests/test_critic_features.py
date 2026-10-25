"""Neutral feature utilities preserve existing frozen critic coordinates."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from math_rl.critic_features import (
    rms_coordinates, isotropic_normalization, hidden_features,
    validate_feature_runtime, FEATURE_PACKAGES,
)


def test_neutral_features_preserve_historical_coordinates():
    # Historical wrappers retain the same exact public transforms.
    from math_rl.direct_jlens import rms_coordinates as old_rms
    from math_rl.direct_jlens import isotropic_normalization as old_normalization
    torch.manual_seed(19)
    x=torch.randn(5,12)
    assert torch.equal(rms_coordinates(x,1e-6),old_rms(x,1e-6))
    assert all(torch.equal(a,b) for a,b in zip(isotropic_normalization(x),old_normalization(x)))
    blocks={2:x,1:x*.3}
    manifest=dict(layers=[2,1],two_thirds=1,rms_eps=1e-6)
    assert torch.equal(hidden_features('rms_hidden_concat',blocks,manifest),
        torch.cat([rms_coordinates(blocks[i],1e-6) for i in [2,1]],1))
    assert torch.equal(hidden_features('hidden_concat',blocks,manifest),torch.cat([x,x*.3],1))
    assert torch.equal(hidden_features('hidden_two_thirds',blocks,manifest),blocks[1])
    with pytest.raises(ValueError,match='Unknown hidden'): hidden_features('invalid',blocks,manifest)
    with pytest.raises(ValueError,match='norm gain'): hidden_features('final',blocks,manifest)


def test_runtime_check_ignores_historical_lens_package(monkeypatch):
    import slice_critic.features as features
    monkeypatch.setattr(features,'version',lambda name:'fixed')
    expected={name:'fixed' for name in FEATURE_PACKAGES}
    expected['jlens']='irrelevant historical package'
    assert validate_feature_runtime(expected)=={name:'fixed' for name in FEATURE_PACKAGES}
    expected['torch']='different'
    with pytest.raises(ValueError,match='runtime changed'): validate_feature_runtime(expected)


def test_active_critic_imports_no_historical_lens_runners():
    root=Path(__file__).resolve().parents[1]
    code="""
import sys
sys.path.insert(0,'scripts')
import sparse_critic_study
import numina_online
import math_rl.bottleneck_features
for name in ('jlens','large_jlens_study','compare_jspace_probes','compress_jlens_critics','math_rl.direct_jlens'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable,'-c',code],cwd=root,check=True,capture_output=True,text=True)


def test_metadata_compatibility_keeps_saved_paths(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from critic_study_io import source_feature_environment
    (tmp_path/'lens-metadata.json').write_text(json.dumps(dict(environment={'torch':'old'})))
    assert source_feature_environment(tmp_path)=={'torch':'old'}
    (tmp_path/'feature-runtime.json').write_text(json.dumps(dict(environment={'torch':'new'})))
    assert source_feature_environment(tmp_path)=={'torch':'new'}
