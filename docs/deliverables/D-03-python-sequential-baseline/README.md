# D-03 - Python sequential baseline

| | |
|---|---|
| **Teacher's item** | Première version fonctionnelle |
| **Depends on** | D-02 |
| **Unblocks** | D-04 |

## What this part is responsible for

The first working end-to-end version: four detectors in plain Python/NumPy/scikit-learn, the quality metrics, and a timing helper. It is the correctness reference the C code must match, and the language baseline (time `T_py`) for the whole comparison.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| NumPy | windows, rolling statistics, median/MAD, distances |
| scikit-learn | Isolation Forest, K-Means fitting, sample scaling |
| pandas | results tables |
| `time.perf_counter` | timers |
| pytest | metric and detector tests |

## Tasks

| Task | What |
|---|---|
| D-03.1 | Window helper and the two statistical detectors (z-score, MAD) in NumPy |
| D-03.2 | Feature extraction `[mean, std, min, max]` |
| D-03.3 | Fit Isolation Forest and K-Means on a sample; compute thresholds |
| D-03.4 | Metrics: precision, recall, F1, event recall |
| D-03.5 | Timing helper and first results table on `syn_1M`, `syn_2M` and two NAB series |

## Pseudo code

```python
W, S = 256, 1

def windows(x, w=W, s=S):
    return np.lib.stride_tricks.sliding_window_view(x, w)[::s]     # a view, no copy

def zscore_py(x, w=W, s=S, chunk=100_000):
    nw = (len(x) - w) // s + 1
    out = np.empty(nw, np.float32)
    for a in range(0, nw, chunk):                                   # chunks keep memory small
        b = min(a + chunk, nw)
        win = windows(x[a*s : (b-1)*s + w], w, s).astype(np.float64)
        mu, sd = win.mean(1), win.std(1, ddof=1)
        last = win[:, -1]
        out[a:b] = np.where(sd > 1e-12, np.abs(last - mu) / sd, 0.0)
    return out

def mad_py(x, w=W, s=S, chunk=50_000):
    ...                                                             # np.median over axis 1, then MAD

def features_py(x, w=W, s=S, chunk=100_000):
    ...                                                             # (nw, 4): mean, std, min, max

def fit_models(F_sample, k=8, seed=0):
    scale = 1.0 / F_sample.std(0)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(F_sample * scale)
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=1).fit(F_sample)
    return km.cluster_centers_ / scale, scale, iso                  # centroids back in raw feature space

def kmeans_score_py(F, centroids, scale):
    d = ((F[:, None, :] - centroids[None, :, :]) * scale) ** 2      # in chunks for large F
    return np.sqrt(d.sum(2).min(1))

def threshold(scores_sample, rho=0.001):
    return np.quantile(scores_sample, 1 - rho)

def evaluate(flags, labels, w=W, s=S, tau=W // 2):
    t = np.arange(len(flags)) * s + (w - 1)                         # sample index of each window
    pred = np.zeros(len(labels), bool); pred[t[flags]] = True
    p, r, f1 = prf(pred, labels)
    return {"precision": p, "recall": r, "f1": f1, "event_recall": event_recall(pred, labels, tau)}
```

Fit sample: 100,000 windows chosen uniformly (fixed seed) from the series. Use the same sample and the same thresholds in every later mode. Note that `centroids / scale` converts centroids fitted in scaled space back to raw feature space, which is what the C kernel expects together with `scale`.

## Files to create

- `python/threadwatch/detectors/{windows.py, zscore.py, mad.py, features.py, ml.py}`
- `python/threadwatch/metrics.py`, `python/threadwatch/bench/timing.py`
- `tests/python/{test_detectors.py, test_metrics.py}`
- `experiments/first_results.py` and `docs/first-results.md` (F1 table and timings)

## Tests and acceptance

- `zscore_py` on a hand-made series with one spike returns a score above 6 at the spike and below 3 elsewhere.
- Metrics tests: perfect flags give F1 = 1; all-zero flags give recall = 0; event recall counts an event once.
- End-to-end run on `syn_1M` and on two NAB series works with no errors.
- A table with F1 (and precision, recall, event recall) for the four detectors is produced.
- `T_py` is recorded for `syn_1M` and `syn_2M`, one row per detector, median of 3 runs.

## Pitfalls

- `sliding_window_view` is a view, but `.mean(1)` on a float32 view can upcast and allocate large temporaries. Process in chunks and accumulate in float64.
- `np.median` over axis 1 is slow and allocates; keep chunks small (tens of thousands of windows).
- Isolation Forest scoring on millions of rows is slow; time it on 1M and 2M only.
- Fit the models once and reuse them; refitting inside the timed region would measure training, not detection.
- Keep `n_jobs=1` for the baseline, otherwise it is not sequential.

## Changelog line

`D-03: Python sequential detectors (z-score, MAD, K-Means, Isolation Forest), metrics, first F1 and T_py table`
