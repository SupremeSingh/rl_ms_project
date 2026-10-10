"""Our sparse reconstruction extension to Anthropic's Jacobian Lens.

The upstream package computes J and vocabulary readouts, not sparse decomposition.
Here nonnegative matching pursuit approximates a J-space component in the original
hidden coordinates. This is not an exact projection or the paper's gradient pursuit.
"""
import torch

JLENS_COMMIT = '581d398613e5602a5af361e1c34d3a92ea82ba8e'
REPRESENTATIONS = ('final', 'two_thirds', 'jspace', 'jspace_final',
                   'two_thirds_final', 'residual', 'random', 'random_final')


def vocabulary_directions(unembedding, norm_weight, jacobian):
    """Qwen RMSNorm gain folded into U; the state-dependent RMS divisor is scalar.

    Rows are directions in *source block-output coordinates*. U diag(g) J h
    agrees with official lens logits up to a positive per-state RMS divisor.
    No final norm is applied to h before reconstruction.
    """
    if unembedding.ndim != 2 or norm_weight.shape != (unembedding.shape[1],):
        raise ValueError('Invalid unembedding/norm dimensions')
    if jacobian.shape != (unembedding.shape[1], unembedding.shape[1]):
        raise ValueError('Invalid Jacobian dimensions')
    directions = (unembedding.float() * norm_weight.float()) @ jacobian.float()
    if not torch.isfinite(directions).all():
        raise ValueError('Nonfinite J-lens directions')
    norms = directions.norm(dim=1, keepdim=True)
    # Zero rows cannot contribute to positive pursuit; keep them zero.
    return directions / norms.clamp_min(1e-12)


def rotated_dictionary(dictionary, seed=913):
    """Fixed signed-coordinate permutation: preserves all atom norms/inner products.

    A geometry-matched random-orientation control, not a Haar-uniform rotation.
    Token labels in this dictionary carry no semantic interpretation.
    """
    rng = torch.Generator(device='cpu').manual_seed(seed)
    order = torch.randperm(dictionary.shape[1], generator=rng).to(dictionary.device)
    signs = (torch.randint(0, 2, (dictionary.shape[1],), generator=rng)*2-1).to(dictionary)
    return dictionary[:, order] * signs


@torch.no_grad()
def reconstruct(hidden, dictionary, steps=16):
    """At most `steps` nonnegative atoms, unit rows, greedy residual line search.

    Repeated atoms are allowed and their positive coefficients add. Each step
    cannot increase squared reconstruction error (up to roundoff). Zero residual
    or nonpositive correlations produce zero additions. Not a global optimum.
    """
    if steps < 1 or hidden.ndim != 2 or hidden.shape[1] != dictionary.shape[1]:
        raise ValueError('Invalid reconstruction shape/budget')
    if not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite hidden features')
    residual = hidden.float().clone()
    result = torch.zeros_like(residual)
    ids, weights = [], []
    for _ in range(steps):
        scores = residual @ dictionary.T
        coefficient, index = scores.max(dim=1)
        coefficient = coefficient.clamp_min(0)
        atom = dictionary[index]
        # Guard against small norm drift in floating point.
        coefficient = coefficient / atom.square().sum(1).clamp_min(1e-12)
        addition = coefficient[:, None] * atom
        result += addition
        residual -= addition
        ids.append(index)
        weights.append(coefficient)
    return result, torch.stack(ids, 1), torch.stack(weights, 1)


def representations(two_thirds, final, jspace, random):
    return dict(final=final, two_thirds=two_thirds, jspace=jspace,
                jspace_final=torch.cat((jspace, final), 1),
                two_thirds_final=torch.cat((two_thirds, final), 1),
                residual=two_thirds-jspace, random=random,
                random_final=torch.cat((random, final), 1))
