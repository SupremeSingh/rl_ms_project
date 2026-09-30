import numpy as np
import pytest
import torch
from math_rl.cumulative_lstd import CumulativeLSTD, raw_design, trajectory_statistics, encoder_epoch


def batch(seed=1):
    gen=torch.Generator().manual_seed(seed)
    features=[torch.randn(3,2,generator=gen) for _ in range(8)]
    masks=torch.ones(8,3);rewards=torch.zeros_like(masks);rewards[:,-1]=torch.arange(8)%2
    return features,rewards,masks,[f'q{i}' for i in range(8)]


@pytest.mark.parametrize('lam',[0.,.99,1.])
def test_statistics_match_explicit_transition_loop(lam):
    states=torch.tensor([[1.,2.],[3.,-1.],[2.,.5]])
    phi=raw_design(states);a=torch.zeros(3,3,dtype=torch.float64);b=torch.zeros(3,dtype=torch.float64);z=torch.zeros(3,dtype=torch.float64)
    for t,p in enumerate(phi):
        z=p+lam*z
        next_phi=phi[t+1] if t+1<len(phi) else torch.zeros_like(p)
        a+=torch.outer(z,p-next_phi)
        if t==len(phi)-1:b+=z
    actual,rhs,n=trajectory_statistics(states,1.,lam)
    torch.testing.assert_close(actual,a);torch.testing.assert_close(rhs,b);assert n==3
    if lam==1:
        torch.testing.assert_close(actual,phi.T@phi)
        torch.testing.assert_close(rhs,phi.sum(0))


def test_raw_cumulative_solve_and_no_repeated_regularization():
    f,r,m,q=batch();c=CumulativeLSTD(q,epsilon=.001,trace_lambda=.99)
    _,_,first=c.update(f,r,m,q,1)
    _,heads,last=c.update(f,r,m,q,2)
    for fold in range(2):
        a=torch.zeros(3,3,dtype=torch.float64);b=torch.zeros(3,dtype=torch.float64)
        for x,y,question in zip(f,r.sum(1),q):
            if c.assignment[question]!=fold:
                ai,bi,_=trajectory_statistics(x,float(y),.99);a+=2*ai;b+=2*bi
        expected=torch.linalg.solve(a+.001*torch.eye(3,dtype=torch.float64),b)
        torch.testing.assert_close(heads[fold]['weights'],expected)
        assert last['folds'][fold]['effective_mean_regularization']==first['folds'][fold]['effective_mean_regularization']/2


def test_historical_question_labels_never_leak():
    f,r,m,q=batch();a=CumulativeLSTD(q,trace_lambda=.99);b=CumulativeLSTD(q,trace_lambda=.99)
    changed=r.clone();held=[i for i,x in enumerate(q) if a.assignment[x]==0]
    changed[held,-1]=1-changed[held,-1]
    for step in (1,2):
        va,_,_=a.update(f,r,m,q,step);vb,_,_=b.update(f,changed,m,q,step)
        torch.testing.assert_close(va[held],vb[held])


def test_refresh_resets_statistics_and_rejects_duplicate_steps():
    f,r,m,q=batch();c=CumulativeLSTD(q,reset_interval=2)
    c.update(f,r,m,q,1);c.update(f,r,m,q,2)
    with pytest.raises(ValueError,match='consecutive'):c.update(f,r,m,q,2)
    v,_,metric=c.update(f,r,m,q,3,feature_epoch=1)
    reference=CumulativeLSTD(q);expected,_,_=reference.update(f,r,m,q,1)
    torch.testing.assert_close(v,expected)
    assert metric['reset'] and sum(x['transitions'] for x in metric['folds'])==24
    assert [encoder_epoch(i,2) for i in range(1,6)]==[0,0,1,1,2]


def test_reward_shape_and_terminal_checks():
    f,r,m,q=batch();c=CumulativeLSTD(q)
    r[0,0]=1
    with pytest.raises(ValueError,match='terminal'):c.update(f,r,m,q,1)


def test_accumulated_ridge_matches_direct_returns_and_lstd_one():
    f,r,m,q=batch()
    ridge=CumulativeLSTD(q,method='ridge')
    lstd=CumulativeLSTD(q,trace_lambda=1.)
    for step in (1,2):
        vr,hr,_=ridge.update(f,r,m,q,step)
        vl,_,_=lstd.update(f,r,m,q,step)
        torch.testing.assert_close(vr,vl)
        for fold in range(2):
            x=torch.cat([raw_design(x) for x,question in zip(f,q) if ridge.assignment[question]!=fold])
            y=torch.cat([torch.full((len(x),),float(y),dtype=torch.float64) for x,y,question in zip(f,r.sum(1),q) if ridge.assignment[question]!=fold])
            expected=torch.linalg.solve(step*x.T@x+.01*torch.eye(x.shape[1]),step*x.T@y)
            torch.testing.assert_close(hr[fold]['weights'],expected)
