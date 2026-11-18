# Which method improves actor accuracy fastest with fresh answers and training compute?

**Not measured yet: the implemented study trains five methods across three fresh seeds for 120 updates, monitors validation accuracy every 10 updates, and evaluates final actors on the shared test set. It will plot mean ± seed SD against updates and measured training GPU-hours, rather than inventing curves from final checkpoints.**

Methods: SLICE/token, SLICE/chunk, PPO, GRPO and RMS-normalized layer/token.
Seeds 1201–1203 give 15 independently scheduled actors, capped at six concurrent
jobs (two GPUs each), with all intermediate actor snapshots retained. Time curves
are hardware-sensitive; the separate controlled benchmark measures matched-workload
overhead.
Monitoring and checkpoint time are measured separately; training GPUh includes
fresh rollout generation. Final test evaluation does not select checkpoints or
tune methods. Same inspected Numina splits and compressor are reused.

[Protocol, resource estimates and commands](../docs/LEARNING_CURVES.md).

No completed results or claim of faster learning, universal stability or
equal-quality training savings yet. Three seeds limit inferential precision.
