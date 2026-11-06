from importlib.resources import files
import torch
from slice_critic.chunks import character_boundaries, ChunkPlan


def test_installed_splitter_preserves_math_and_includes_license():
    text='Expand the expression $x^2+2x+1$. Then simplify the result.\nFinish with the final answer.'
    points=character_boundaries(text,'math')
    assert ''.join(text[a:b] for a,b in zip(points,points[1:]))==text
    assert len(points)>2
    license=files('slice_critic._vendor').joinpath('VINEPPO_LICENSE').read_text()
    assert 'Copyright (c) 2024 McGill NLP' in license


def test_broadcast_never_drops_terminal_action():
    p=ChunkPlan((0,2),(2,5),5)
    assert p.expand(torch.tensor([.2,.5])).tolist()==[float(torch.tensor(.2))]*2+[.5]*3
