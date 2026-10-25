# Internal changelog

## 0.1.1 — internal candidate, 25 October 2026

- Public artifact inspection and PPO feature export/loading/readout interfaces.
- PPO uses public interfaces; previous compatibility imports remain supported.
- Validate artifact metadata, tensor dimensions, calibration and prefix inputs before inference.
- Installed-wheel CPU acceptance outside the checkout, with research imports blocked.
- CI on Python 3.10 and 3.12; explicit public API and release procedure.
- Completed v1 artifacts remain supported. No Ridge/LSTD/PPO equations changed.

This is not a PyPI publication. Check CI and cluster integration for the revision
being consumed before distributing its wheel internally.

## 0.1.0 — initial internal extraction

- Frozen multi-layer features, sparse/plain compressors, Ridge/LSTD and local scoring.
- Checksummed artifacts, stage recovery and isolated Math-Verify interpreter.
- Real-model standalone smoke and two-update Ridge/LSTD PPO smoke passed on Duke.
