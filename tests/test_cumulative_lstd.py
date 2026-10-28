import numpy as np
import pytest
import torch
from slice_critic.cumulative import CumulativeCritic, raw_design, trajectory_statistics, encoder_epoch


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
    f,r,m,q=batch();c=CumulativeCritic(q,epsilon=.001,trace_lambda=.99,folds=2)
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
    f,r,m,q=batch();a=CumulativeCritic(q,trace_lambda=.99,folds=2);b=CumulativeCritic(q,trace_lambda=.99,folds=2)
    changed=r.clone();held=[i for i,x in enumerate(q) if a.assignment[x]==0]
    changed[held,-1]=1-changed[held,-1]
    for step in (1,2):
        va,_,_=a.update(f,r,m,q,step);vb,_,_=b.update(f,changed,m,q,step)
        torch.testing.assert_close(va[held],vb[held])


def test_refresh_resets_statistics_and_rejects_duplicate_steps():
    f,r,m,q=batch();c=CumulativeCritic(q,reset_interval=2,folds=2)
    c.update(f,r,m,q,1);c.update(f,r,m,q,2)
    with pytest.raises(ValueError,match='consecutive'):c.update(f,r,m,q,2)
    v,_,metric=c.update(f,r,m,q,3,feature_epoch=1)
    reference=CumulativeCritic(q,folds=2);expected,_,_=reference.update(f,r,m,q,1)
    torch.testing.assert_close(v,expected)
    assert metric['reset'] and sum(x['transitions'] for x in metric['folds'])==24
    assert [encoder_epoch(i,2) for i in range(1,6)]==[0,0,1,1,2]


def test_reward_shape_and_terminal_checks():
    f,r,m,q=batch();c=CumulativeCritic(q,folds=2)
    r[0,0]=1
    with pytest.raises(ValueError,match='terminal'):c.update(f,r,m,q,1)


def test_accumulated_ridge_matches_direct_returns_and_lstd_one():
    f,r,m,q=batch()
    ridge=CumulativeCritic(q,method='ridge',folds=2)
    lstd=CumulativeCritic(q,trace_lambda=1.,folds=2)
    for step in (1,2):
        vr,hr,_=ridge.update(f,r,m,q,step)
        vl,_,_=lstd.update(f,r,m,q,step)
        torch.testing.assert_close(vr,vl)
        for fold in range(2):
            x=torch.cat([raw_design(x) for x,question in zip(f,q) if ridge.assignment[question]!=fold])
            y=torch.cat([torch.full((len(x),),float(y),dtype=torch.float64) for x,y,question in zip(f,r.sum(1),q) if ridge.assignment[question]!=fold])
            expected=torch.linalg.solve(step*x.T@x+.01*torch.eye(x.shape[1]),step*x.T@y)
            torch.testing.assert_close(hr[fold]['weights'],expected)


@pytest.mark.parametrize('method,lam',[('ridge',.9),('lstd',0.),('lstd',.9),('lstd',1.)])
def test_one_system_matches_all_data_explicit_equations_across_updates(method,lam):
    f,r,m,q=batch()
    critic=CumulativeCritic(q,method=method,trace_lambda=lam,folds=1)
    a=torch.zeros(3,3,dtype=torch.float64);b=torch.zeros(3,dtype=torch.float64)
    for step in (1,2):
        features,rewards,mask,questions=batch(seed=step)
        for features_i,outcome in zip(features,rewards.sum(1).tolist()):
            phi=raw_design(features_i);z=torch.zeros(3,dtype=torch.float64)
            for t,x in enumerate(phi):
                following=phi[t+1] if t+1<len(phi) else torch.zeros_like(x)
                if method=='ridge':
                    a+=torch.outer(x,x);b+=x*outcome
                else:
                    z=lam*z+x
                    a+=torch.outer(z,x-following)
                    b+=z*(outcome if t==len(phi)-1 else 0.)
        values,heads,metrics=critic.update(features,rewards,mask,questions,step)
        weights=torch.linalg.solve(a+.01*torch.eye(3,dtype=torch.float64),b)
        assert len(critic.stats)==len(heads)==1
        torch.testing.assert_close(critic.stats[0]['a'],a)
        torch.testing.assert_close(critic.stats[0]['b'],b)
        torch.testing.assert_close(heads[0]['weights'],weights)
        torch.testing.assert_close(values,torch.stack([(raw_design(x)@weights).float() for x in features]))
        assert critic.stats[0]['answers']==8*step and critic.stats[0]['transitions']==24*step
        assert not metrics['cross_fitted'] and metrics['critic_folds']==1
        assert 'in_sample_brier' in metrics and 'out_of_fold_brier' not in metrics
        assert heads[0]['predicted_fold']==heads[0]['fitting_fold']==0
    assert set(critic.state_dict()['assignment'].values())=={0}
    assert critic.state_dict()['folds']==1


def test_one_head_really_uses_current_rewards_and_does_not_require_four_questions():
    f,r,m,q=batch()
    q=['same question']*len(q)
    a=CumulativeCritic(q,method='ridge',folds=1)
    b=CumulativeCritic(q,method='ridge',folds=1)
    va,_,_=a.update(f,r,m,q,1)
    vb,_,_=b.update(f,torch.zeros_like(r),m,q,1)
    assert not torch.allclose(va,vb)
    assert torch.equal(vb,torch.zeros_like(vb))


@pytest.mark.parametrize('folds',[0,3,True,1.0,'1'])
def test_invalid_fold_modes_rejected(folds):
    with pytest.raises(ValueError,match='folds'):
        CumulativeCritic(['a','b','c','d'],folds=folds)
