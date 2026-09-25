# Prediction Error Ordering in Learning-Augmented Online Algorithms

Experiment code and measured results behind the research proposal *A Study of
Prediction Error Ordering in Learning-Augmented Online Algorithms* (SIT320
Advanced Algorithms, Deakin University).

## The question

Learning-augmented analyses of Online Facility Location (OFL) bound the
competitive ratio in terms of scalar error measures — typically the total error
$\eta_1$ and the largest single error $\eta_\infty$. Both are permutation
invariant: reshuffle the error sequence and neither number moves.

The experiments here ask whether the cost moves. It does.

## What is in the repository

Only the code and data that the proposal actually reports. The wider set of
exploratory runs from the study is not included.

```
src/
  pipeline.py            Trace loading, the three crude forecasters, error series,
                         autocorrelation, burst lengths, integrated timescale
  ofl_cost2.py           OFL instance construction and the Threshold(θ) algorithm
  robust_c1.py           The Meyerson randomised rule
  c1_final.py            The Trust(ε) rule and the per-series statistics
  verify_c1.py           Main experiment: 900 error series, real order vs its own
                         permutations, with a shuffled-series control
  order_control.py       Controlled-ordering experiment: the error multiset is held
                         fixed and only the order is imposed
  spectrum_scan.py       Spectral scan for the dominant period of each error series
  separation_example.py  The constructed instance, checked against the same simulator
  fig_verify.py          Aggregates the two headline tables (figH, figK)

results/
  verify_c1_d{01,02,10}.csv        Per-series outcome, 900 series over 3 days
  verify_c1_summary.txt            Aggregate of the above
  order_control_d{01,02,10}.csv    Per-series cost under each imposed order
  order_control_summary.txt        Aggregate of the above
  figH_direction.csv               Real order vs permutations, per algorithm
  figK_counterexample.csv          Cost of seven orderings of one fixed multiset
  d{01,02}/spectrum_period_histogram.csv   Dominant-period histogram
```

## Results the proposal reports

**A constructed instance.** `separation_example.py` builds $n$ clients at the
point $0$ with an error multiset of $n/2$ zeros and $n/2$ copies of $M$. Both
orderings give $\eta_1 = (n/2)M$, $\eta_\infty = M$ and $\mathrm{OPT} = f$, yet
the costs are $2f$ and $2f + (n/2)M$ — a ratio of $1 + nM/(4f)$, unbounded in
both $n$ and $M$. The script confirms the hand calculation against the same
simulator the measurements use.

**Real error orderings are structured.** The dominant periods concentrate at 3,
5, 10, 15, 30 and 60 minutes — the marks people pick when setting automated
schedules — and errors run in episodes longer than independent per-minute errors
would produce (`spectrum_period_histogram.csv`).

**The real order is systematically cheaper than its own permutations**
(`figH_direction.csv`, 900 series, 60 permutations each, common random numbers):

| Algorithm | Median gap | Share cheaper than shuffled |
|---|---|---|
| Threshold(θ) | −3.86 % | 69.3 % |
| Meyerson | −0.71 % | 64.1 % |
| Trust(ε) | −0.60 % | 66.0 % |

The shuffled-series control sits near 53–57 %, so the effect is not an artefact
of the comparison.

**Ordering alone moves the cost on real data** (`figK_counterexample.csv`, 450
series, one fixed multiset per series, seven orderings). Descending against
ascending order:

| Algorithm | Cost ratio |
|---|---|
| Threshold(θ) | 2.38 |
| Meyerson | 1.76 |
| Trust(ε) | 1.33 |

$\eta_1$ and $\eta_\infty$ are identical for every column. The ordering the
production workload produces sits near the cheap end; the expensive end is
reached only by deliberate sorting.

## Reproducing

Python 3.11 with `numpy`, `pandas`, `scipy` and `matplotlib`.

The input is the Azure Functions 2019 trace, which is not in this repository
(roughly 590 MB). Download `invocations_per_function_md.anon.d01.csv`, `.d02.csv`
and `.d10.csv` from [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)
into `data/`, then:

```bash
python src/separation_example.py             # the constructed instance
python src/spectrum_scan.py --day 1          # dominant periods, repeat for --day 2
python src/verify_c1.py --days 1 2 10 --procs 8      # the 900-series experiment
python src/order_control.py --days 1 2 10 --procs 8  # controlled ordering
python src/fig_verify.py                             # aggregate figH and figK
```

`verify_c1.py` and `order_control.py` use `multiprocessing`. Keep `--procs`
modest: each worker holds its own slice of the trace, and on a 16 GB machine a
higher count exhausted memory. The runs are long, which is why the
seven-ordering experiment covers 450 series rather than all 900. Every run is seeded, and the randomised algorithms are compared on
common random numbers so that a cost difference cannot come from the luck of a
single draw.

## Scope

These are preliminary results supporting a proposal, not a finished study. They
rest on one simulation model and one data source, and the constructed separation
applies to algorithms that place facilities at the predicted location. The
proposal states the accompanying limitations.

## Data source

Microsoft, *Azure Functions trace 2019*, AzurePublicDataset —
https://github.com/Azure/AzurePublicDataset
