"""Frozen-feature value probes. No actor training and no TD/LSTD fitting here."""
import math
import time

import numpy as np
import torch
from torch import nn

HEADS = ("linear_logit", "mlp2", "resnet10", "linear_value")
# Fixed before observing a trajectory's length or outcome. Include only reached states.
PREFIX_POSITIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x) / 2


def make_head(kind, dimension, width=256):
    if kind in ("linear_logit", "linear_value"):
        return nn.Linear(dimension, 1)
    if kind == "mlp2":
        return nn.Sequential(nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, 1))
    if kind == "resnet10":
        return nn.Sequential(nn.Linear(dimension, width), nn.GELU(),
                             *(ResidualBlock(width) for _ in range(4)), nn.Linear(width, 1))
    raise ValueError(f"Unknown head: {kind}")


def prefix_states(backbone, token_ids, prompt_length, response_length):
    """T+1 post-normalization vectors: pre-action states 0..T-1 and terminal context T."""
    if prompt_length < 1 or response_length < 1 or token_ids.shape[1] != prompt_length + response_length:
        raise ValueError("Invalid prompt/response boundary")
    with torch.inference_mode():
        hidden = backbone(input_ids=token_ids, attention_mask=torch.ones_like(token_ids),
                          use_cache=False, return_dict=True).last_hidden_state
        states = hidden[0, prompt_length - 1:prompt_length + response_length].detach().cpu().clone()
    if states.shape[0] != response_length + 1 or not torch.isfinite(states).all():
        raise ValueError("Incomplete or nonfinite prefix features")
    return states


def transitions(states, reward):
    """LSTD-ready alignment; the last response action receives R and ends the episode."""
    count = len(states) - 1
    if count < 1 or reward not in (0, 1):
        raise ValueError("Invalid trajectory")
    rewards = torch.zeros(count)
    rewards[-1] = reward
    done = torch.zeros(count, dtype=torch.bool)
    done[-1] = True
    following = states[1:].clone()
    following[-1] = 0  # Never bootstrap from post-terminal context, including at the cap.
    return states[:-1], following, rewards, done


def sample_shard(shard, question_index):
    xs, ys, positions = [], [], []
    for i, reward in enumerate(shard["rewards"]):
        start, stop = shard["offsets"][i:i + 2].tolist()
        length = stop - start - 1
        selected = [p for p in PREFIX_POSITIONS if p < length]
        if not selected:
            raise ValueError("Empty trajectory")
        xs.append(shard["features"][start + torch.tensor(selected)])
        ys.extend([float(reward)] * len(selected))
        positions.extend(selected)
    x = torch.cat(xs)
    return dict(x=x, y=torch.tensor(ys), position=torch.tensor(positions),
                question=torch.full((len(ys),), question_index, dtype=torch.long))


def auc(y, prediction):
    positive = y == 1
    npos, nneg = positive.sum(), (~positive).sum()
    if not npos or not nneg:
        return None
    order = np.argsort(prediction, kind="stable")
    _, start, counts = np.unique(prediction[order], return_index=True, return_counts=True)
    ranks = np.empty(len(y))
    ranks[order] = np.repeat(start + (counts + 1) / 2, counts)
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not len(y) or y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Empty, mismatched or nonfinite predictions")
    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    calibration = []
    bins = np.minimum((np.clip(prediction, 0, 1) * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        if mask.any():
            calibration.append(dict(count=int(mask.sum()), prediction=float(prediction[mask].mean()),
                                    success_rate=float(y[mask].mean())))
    return dict(n=len(y), accuracy=float(((prediction >= .5) == y).mean()),
                brier=float(np.mean((prediction - y) ** 2)),
                cross_entropy=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                auroc=auc(y, prediction), outside_probability_range=float(((prediction < 0) | (prediction > 1)).mean()),
                calibration=calibration,
                ece=sum(b["count"] * abs(b["prediction"] - b["success_rate"]) for b in calibration) / len(y))


def assessment(data, prediction, baseline):
    y, q, pos = (data[k].numpy() for k in ("y", "question", "position"))
    result = metrics(y, prediction)
    result["by_prefix"] = {name: metrics(y[mask], prediction[mask]) for name, mask in
        {"question_only": pos == 0, "early_1_32": (pos > 0) & (pos <= 32),
         "middle_33_128": (pos > 32) & (pos <= 128), "late_129_plus": pos > 128}.items() if mask.any()}
    # Cluster uncertainty by question: all its responses and prefixes move together.
    losses = (prediction - y) ** 2 - (baseline - y) ** 2
    _, group = np.unique(q, return_inverse=True)
    per_question = np.bincount(group, weights=losses) / np.bincount(group)
    rng = np.random.default_rng(42)
    samples = [rng.choice(per_question, len(per_question), replace=True).mean() for _ in range(1000)]
    result["question_mean_brier_difference_vs_constant"] = float(per_question.mean())
    result["question_bootstrap_95pct_interval"] = np.quantile(samples, [.025, .975]).tolist()
    return result


def predict(head, kind, data, mean, scale, device, batch_size=512):
    head.eval()
    values = []
    with torch.inference_mode():
        for batch in data["x"].split(batch_size):
            output = head((batch.to(device).float() - mean) / scale).flatten()
            values.append((output if kind == "linear_value" else output.sigmoid()).cpu())
    return torch.cat(values).numpy()


def fit_trial(kind, train, val, mean, scale, device, seed, lr, epochs, width=256):
    torch.manual_seed(seed)
    head = make_head(kind, train["x"].shape[1], width).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss() if kind == "linear_value" else nn.BCEWithLogitsLoss()
    rng = torch.Generator().manual_seed(seed)
    best, best_state, curve = math.inf, None, []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        head.train()
        for indices in torch.randperm(len(train["y"]), generator=rng).split(512):
            x, y = train["x"][indices].to(device).float(), train["y"][indices].to(device)
            loss = criterion(head((x - mean) / scale).flatten(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {kind}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        prediction = predict(head, kind, val, mean, scale, device)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Nonfinite validation predictions in {kind}")
        error = float(np.mean((prediction - val["y"].numpy()) ** 2))
        curve.append(dict(epoch=epoch + 1, seconds=time.perf_counter() - started, validation_brier=error))
        if error < best:
            best = error
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    return dict(state=best_state, kind=kind, seed=seed, lr=lr, width=width, validation_brier=best,
                curve=curve, fitting_seconds=time.perf_counter() - started,
                parameters=sum(p.numel() for p in head.parameters()),
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
