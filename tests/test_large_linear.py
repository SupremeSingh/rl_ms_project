import torch
import pytest
from math_rl.large_linear import accumulate, solve, statistics
from math_rl.lstd import design


def test_trace_matrices_match_explicit_recurrence_and_terminal_reward():
    torch.manual_seed(51)
    x = torch.randn(7, 3); mean = torch.zeros(3); scale = torch.ones(3)
    stats = statistics(3); accumulate(stats, x, 1, mean, scale, chunk=2)
    phi = design(x, mean, scale); z = torch.zeros(4, dtype=torch.float64)
    a = torch.zeros(4, 4, dtype=torch.float64); b = torch.zeros(4, dtype=torch.float64)
    for i in range(len(x)):
        z = .99*z+phi[i]
        following = phi[i+1] if i+1 < len(x) else torch.zeros(4)
        a += torch.outer(z, phi[i]-following)
        b += z*(i == len(x)-1)
    torch.testing.assert_close(stats['a'], a)
    torch.testing.assert_close(stats['b'], b)
    torch.testing.assert_close(stats['gram'], phi.T@phi)
    torch.testing.assert_close(stats['returns_rhs'], phi.sum(0))
    assert stats['transitions'] == 7


def test_trace_resets_per_answer_and_shard_sums_agree():
    torch.manual_seed(7)
    mean, scale = torch.zeros(2), torch.ones(2)
    x, y = torch.randn(3, 2), torch.randn(5, 2)
    left, right, both = [statistics(2) for _ in range(3)]
    accumulate(left, x, 0, mean, scale); accumulate(right, y, 1, mean, scale)
    accumulate(both, x, 0, mean, scale); accumulate(both, y, 1, mean, scale)
    for key in both:
        if torch.is_tensor(both[key]):
            torch.testing.assert_close(both[key], left[key]+right[key])
        else:
            assert both[key] == left[key]+right[key]


def test_lambda_one_equals_ridge_and_raw_epsilon_is_not_rescaled():
    torch.manual_seed(12)
    stats = statistics(2); mean, scale = torch.zeros(2), torch.ones(2)
    for reward in (0, 1, 1):
        accumulate(stats, torch.randn(8, 2), reward, mean, scale, trace_lambda=1, chunk=3)
    torch.testing.assert_close(stats['a'], stats['gram'])
    torch.testing.assert_close(stats['b'], stats['returns_rhs'])
    ridge, _ = solve(stats, 'ridge', .01)
    lstd, checks = solve(stats, 'lstd-0.99', .01)
    torch.testing.assert_close(lstd, ridge)
    expected = torch.linalg.solve(stats['a']+.01*torch.eye(3), stats['b'])
    torch.testing.assert_close(lstd, expected)
    assert checks['relative_residual'] < 1e-8
    with pytest.raises(ValueError):
        solve(stats, 'ridge', -1)
