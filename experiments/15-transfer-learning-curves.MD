# Does transferred SLICE help when target-domain labels are scarce?

**The clearest gain was MATH with 100 training questions: transferred SLICE Brier 0.219051 versus RMS-layer 0.228839, difference interval [−0.01490, −0.00445]. At 2,000 questions it was essentially tied on MATH and worse on GSM8K, so the low-data benefit was task-dependent.**

| Target | Training questions | RMS layer Brier | Transferred SLICE Brier | Target-trained SLICE Brier |
|---|---|---|---|---|
| GSM8K | 100 | 0.233105 | 0.230942 | 0.240831 |
| GSM8K | 500 | 0.207439 | 0.208633 | 0.217556 |
| GSM8K | 2000 | 0.199678 | 0.203674 | 0.215138 |
| MATH | 100 | 0.228839 | 0.219051 | 0.216711 |
| MATH | 500 | 0.195342 | 0.193417 | 0.201400 |
| MATH | 2000 | 0.185528 | 0.185644 | 0.196631 |

Offline feature transfer only: all heads refit on target labels, with five representation/adaptation seeds 42–46. The RMS baseline is deterministic for a fixed question subset, so repeated identical baseline results are not five independent encoder fits. Training budgets are questions, with four answers/question—not merely 100 individual prefixes. Evaluation gives each question equal weight.

GSM8K used 1,000 test questions; MATH retained 294 eligible/disjoint test questions from the intended MATH500 source, not the complete public MATH500 benchmark. Validation/test collections and exclusions are detailed in the cost ledger. Models and penalties were locked before final test evaluation. The expanded grid included .01, 1,000 and 10,000; the supplied validation selections mostly chose 1,000. No test-based reselection.

Transferred SLICE minus RMS intervals: GSM8K/100 [−0.00530, 0.00070], /500 [−0.00107, 0.00336], /2000 [0.00223, 0.00573]; MATH/100 [−0.01490, −0.00445], /500 [−0.00571, 0.00192], /2000 [−0.00313, 0.00329]. These are exploratory 95% intervals, not proof of universal few-shot superiority. Transferred features usually outperformed learning a new target compressor at these budgets.

## Evidence and scope

Cluster: `outputs/transfer-clarify-v1/validation-report.txt` and `report.txt`, supplied in the project discussion. [Independent cost/data ledger extract](data/cost-report.txt). Prefix prediction, not actor accuracy or policy transfer.
