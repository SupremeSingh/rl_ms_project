# Internal release notes

## 0.3.0 - one-head cumulative critic experiment

- Default `folds=1` shared-head Ridge/LSTD: accumulate all answers, solve once,
  then predict the current batch. Explicit `folds=2` remains available for historical reproduction.
- Explicit system routing, fold counts and in-sample versus out-of-fold metrics
  in checkpoints/diagnostics. Offline artifact schema and equations unchanged.
- New no-fold six-method protocol: one head in every linear arm, original
  data/filter/budgets and method comparisons, mandatory GPU smoke before training.

## 0.2.0 — final workflow cleanup, 25 October 2026

- One authoritative implementation for feature transforms, grading, numeric
  references, question families, folds and linear fitting.
- Public frozen raw-layer readout for matched controls; cumulative class named
  `CumulativeCritic` for both Ridge and LSTD.
- Removed unused research helpers and consuming-project compatibility wrappers.
- Completed v1 artifacts remain readable; unfinished stage recovery still rejects
  changed source/configuration. Ridge/LSTD equations are unchanged.
- The research consumer now has one six-method protocol and a portable final-input
  capture command, plus a separate from-scratch preparation route.

Internal only. The refactored consumer requires a fresh cluster smoke before a
new full experiment. Earlier standalone and PPO integration checks passed before
this structural cleanup; they do not certify the changed revision automatically.
