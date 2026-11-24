"""Validation-only probability calibration and basic prediction diagnostics."""
import torch
import torch.nn.functional as F


def calibrate(raw, labels):
    raw=torch.as_tensor(raw,dtype=torch.float64);labels=torch.as_tensor(labels,dtype=torch.float64)
    if set(labels.tolist())!={0.,1.}: raise ValueError('Calibration requires both outcome classes')
    coefficients=torch.tensor([1.,0.],dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([coefficients],max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=F.binary_cross_entropy_with_logits(coefficients[0]*raw+coefficients[1],labels)+1e-5*coefficients.square().sum()
        loss.backward();return loss
    optimizer.step(closure)
    if not torch.isfinite(coefficients).all(): raise ValueError('Nonfinite calibration fit')
    return dict(method='platt',slope=float(coefficients[0].detach()),intercept=float(coefficients[1].detach()),
                scope='Validation-only calibration; raw values remain available separately')


def metrics(labels, raw, calibration, constant):
    y=torch.as_tensor(labels).double();raw=torch.as_tensor(raw).double()
    if not len(y) or not torch.isfinite(raw).all(): raise ValueError('Empty or nonfinite predictions')
    probability=torch.sigmoid(raw*calibration['slope']+calibration['intercept'])
    return dict(prefixes=len(y),success_rate=float(y.mean()),raw_brier=float((raw-y).square().mean()),
                calibrated_brier=float((probability-y).square().mean()),
                accuracy=float(((probability>=.5)==y.bool()).double().mean()),
                constant_brier=float((y-constant).square().mean()),
                raw_outside_unit_interval=float(((raw<0)|(raw>1)).double().mean()))
