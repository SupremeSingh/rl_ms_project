"""Single-owner neural value critic. Fresh rollouts only; no cumulative matrices.

Values precede GAE; detached GAE returns subsequently fit this critic. Joint
features are cached frozen-encoder coordinates, never changing compressor outputs.
"""
import copy
import time
import numpy as np
import torch
from torch import nn


class NeuralCritic(nn.Module):
    def __init__(self, fitted, *, joint=False, seed=42, epochs=2, head_lr=3e-4,
                 compressor_lr=3e-6, minibatch=16, chunk=1024, device='cpu'):
        super().__init__()
        compressor,mean,scale=fitted
        self.joint=joint
        self.epochs,self.minibatch,self.chunk=epochs,minibatch,chunk
        self.compressor=copy.deepcopy(compressor).cpu().requires_grad_(joint)
        self.compressor.readout.requires_grad_(False)
        self.register_buffer('mean',mean.cpu().clone())
        self.register_buffer('scale',scale.cpu().clone())
        with torch.random.fork_rng(devices=[]):
            # CPU initialization only: do not reseed the actor worker's CUDA RNG.
            torch.random.default_generator.manual_seed(seed)
            self.head=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Linear(256,1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        self.to(device)  # Move before constructing the optimizer.
        groups=[dict(params=self.head.parameters(),lr=head_lr)]
        if joint: groups.append(dict(params=self.compressor.projection.parameters(),lr=compressor_lr))
        self.optimizer=torch.optim.AdamW(groups,weight_decay=.001)
        self.generator=torch.Generator().manual_seed(seed)
        self.step=0

    def forward(self,x):
        z=self.compressor.features((x-self.mean)/self.scale) if self.joint else x
        return self.head(z).flatten().sigmoid()

    @property
    def device(self):
        return self.head[0].weight.device

    def inputs(self,features,mask):
        if mask.ndim!=2: raise ValueError('Expected token masks')
        lengths=mask.detach().cpu().sum(-1).long()
        expected=torch.arange(mask.shape[1])[None,:]<lengths[:,None]
        if (lengths<=0).any() or not torch.equal(mask.detach().cpu().bool(),expected):
            raise ValueError('Expected nonempty contiguous masks')
        width=7680 if self.joint else 768
        xs=[torch.as_tensor(np.asarray(x)).detach().cpu().float() for x in features]
        if len(xs)!=len(lengths) or any(x.shape!=(int(n),width) or not torch.isfinite(x).all()
                                       for x,n in zip(xs,lengths)):
            raise ValueError('Neural feature/mask mismatch')
        return xs

    @torch.inference_mode()
    def values(self,xs,mask):
        values=torch.zeros(mask.shape,dtype=torch.float32)
        for i,x in enumerate(xs):
            values[i,:len(x)]=torch.cat([self(v.to(self.device)).cpu() for v in x.split(self.chunk)])
        if not torch.isfinite(values).all(): raise ValueError('Nonfinite neural values')
        return values

    @torch.enable_grad()
    def fit_returns(self,xs,returns,old_values,mask,step):
        if step!=self.step+1: raise ValueError('Neural updates must be consecutive')
        targets=returns.detach().cpu().float()
        old=old_values.detach().cpu().float()
        if targets.shape!=mask.shape or old.shape!=mask.shape or not torch.isfinite(targets).all():
            raise ValueError('Invalid detached GAE targets')
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        before=self.values(xs,mask)
        def mse(values):
            return float(torch.stack([(values[i,:len(x)]-targets[i,:len(x)]).square().mean()
                                      for i,x in enumerate(xs)]).mean())
        total_updates=0
        for _ in range(self.epochs):
            for ids in torch.randperm(len(xs),generator=self.generator).split(self.minibatch):
                self.optimizer.zero_grad(set_to_none=True)
                for i in ids.tolist():
                    for offset in range(0,len(xs[i]),self.chunk):
                        x=xs[i][offset:offset+self.chunk].to(self.device)
                        value=self(x)
                        target=targets[i,offset:offset+len(x)].to(self.device)
                        prior=old[i,offset:offset+len(x)].to(self.device)
                        clipped=prior+(value-prior).clamp(-.5,.5)
                        loss=.5*torch.maximum((value-target).square(),(clipped-target).square()).sum()/len(xs[i])/len(ids)
                        if not torch.isfinite(loss): raise ValueError('Nonfinite value loss')
                        loss.backward()
                norm=nn.utils.clip_grad_norm_([p for p in self.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm): raise ValueError('Nonfinite neural gradients')
                self.optimizer.step()
                total_updates+=1
        after=self.values(xs,mask)
        self.step=step
        return dict(step=step,backend=self.device.type,joint=self.joint,buffer_capacity=0,
                    cumulative_matrices=False,target='detached_fresh_GAE_returns',
                    mse_before=mse(before),mse_after=mse(after),optimizer_steps=total_updates,
                    seconds=time.perf_counter()-start,answers=len(xs),
                    feature_bytes=sum(x.numel()*x.element_size() for x in xs),
                    process_peak_cuda_bytes=(torch.cuda.max_memory_allocated(self.device)
                                             if self.device.type=='cuda' else None))


class NeuralCriticService:
    """Two-phase RPC state machine owned by one GPU worker, not the Ray driver.

    Only one rollout is retained between predict and fit. Old values and masks
    stay paired with that rollout; the subsequent fit never resends raw features.
    CPU device is supported for unit tests only; production rejects CPU replies.
    """
    def __init__(self,fitted,*,device,**settings):
        self.critic=NeuralCritic(fitted,device=device,**settings)
        self.pending=None

    def predict(self,features,mask,step):
        if self.pending is not None or step!=self.critic.step+1:
            raise ValueError('A new rollout requires the previous fit to complete')
        mask=mask.detach().cpu().clone()
        xs=self.critic.inputs(features,mask)
        values=self.critic.values(xs,mask).clone()
        self.pending=(xs,mask,values.clone(),step)
        return dict(values=values,backend=self.critic.device.type,owner_rank=0,step=step)

    def fit(self,returns,mask,step,checkpoint=False):
        if self.pending is None: raise ValueError('Predict before fitting')
        xs,expected,old,prediction_step=self.pending
        if step!=prediction_step or not torch.equal(mask.detach().cpu(),expected):
            raise ValueError('Fit is not paired with the predicted rollout')
        metrics=self.critic.fit_returns(xs,returns,old,expected,step)
        metrics.update(owner_rank=0,frozen_encoder=True)
        state={k:v.detach().cpu().clone() for k,v in self.critic.state_dict().items()} if checkpoint else None
        self.pending=None  # No replay across updates.
        return dict(metrics=metrics,state=state,backend=self.critic.device.type,owner_rank=0,step=step)
