# fastcmb

fastcmb is an online emulator for Boltzmann codes, aimed at CMB analyses.
It sits in front of CAMB, or any other code, and works one point at a
time:

- It predicts the spectra together with the error of that prediction,
  measured in units of `ln L`.
- If the predicted error is below the tolerance, it returns the
  prediction. This takes about 10 ms.
- Otherwise it runs the code and adds the point to its training set.

There is nothing to train in advance: the emulator learns while the
chain runs.

The package itself depends only on numpy and scipy. Three integrations
are available:

| Where | How |
|---|---|
| **CosmoFit** | `theory: {background: ..., fastcmb: ...}` in place of `camb` (built in, with `pip install "cosmofit[fast]"`) |
| **cobaya** | the theory `fastcmb.cobaya.FastCAMB` in place of `camb` |
| **anything else** | call a `CMBEmulator` |

## Install

```bash
pip install -e ".[camb]"     # with CAMB as the oracle
pip install -e ".[cobaya]"   # and the cobaya theory
```

## Use

### Python

```python
from fastcmb import CAMBOracle, CMBEmulator

emulator = CMBEmulator(CAMBOracle(lmax=2508, lens_potential_accuracy=4),
                       tolerance=0.05, training_file="emu/lcdm")

result = emulator({"H0": 67.4, "ombh2": 0.0224, "omch2": 0.120,
                   "ln1e10As": 3.044, "ns": 0.965, "tau": 0.054})

result.Cl["tt"], result.Cl["pp"], result.sigma8, result.emulated
```

`result` contains the following:

- `Cl` holds `ell`, `tt`, `te`, `ee` and `bb`, starting at `l = 0`. These
  are `C_l` in muK².
- `pp` is `[L(L+1)]² C_L / 2π`.

Any code can serve as the oracle. It is a function that takes a
parameter dict and returns `{"Cl": ..., "sigma8": ..., "sigma8_cb": ...}`
in these conventions, or `None` when the point cannot be computed.

### cobaya

```yaml
theory:
  fastcmb.cobaya.FastCAMB:
    tolerance: 0.05
    lens_potential_accuracy: 4
    training_file: chains/emu/lcdm
params:
  logA: ...     # or As
  ns: ...
  H0: ...       # or cosmomc_theta
  ombh2: ...
  omch2: ...
  tau: ...
  sigma8:       # derived
```

The theory provides `Cl` in cobaya's units and `ell_factor` conventions,
so cobaya's CMB likelihoods read it as they would read `camb`.

### CosmoFit

```yaml
theory:
  background:
  fastcmb:
    tolerance: 0.05
    training_file: chains/emu/lcdm
```

The theory takes every `camb` option and parameter.

## Options

| | default | |
|---|---|---|
| `tolerance` | 0.05 | largest predicted `ln L` error the emulator answers with |
| `sigma8_tolerance` | 1e-3 | largest predicted relative error of `sigma8` |
| `min_train` | 30 | points computed by the code before the first fit |
| `max_train` | 800 | most points fitted on (the latest) |
| `training_file` | none | keeps the points, so parallel chains and later runs share them |
| `check_every` | 200 | every N predictions the code runs as well, to measure the error |

## How it works

1. **Transformation.**
   - `tt`, `ee`, `bb` and `pp` are emulated as `ln C_l`.
   - `te` is emulated as the correlation `C_TE / sqrt(C_TT C_EE)`.
2. **Compression.** Each block is reduced with PCA. The number of
   components is chosen so that the PCA error in `ln L` stays below 0.01
   (`compress.py`).
3. **Gaussian process.**
   - The mean is quadratic in the inputs.
   - The four components that carry the most likelihood each get their
     own kernel, and the remaining components share one (`gp.py`).
   - Hyperparameters are fitted by maximum likelihood on 200 of the
     points. Each refit tries both the previous values and a fresh
     start, and it reruns whenever half the training set is new.
4. **Error.**
   - Each kernel's predictive variance is projected onto `ln L` with an
     idealized Planck Fisher weight (`metric.py`).
   - Variances are computed in order of their bound and only while they
     can still change the decision. This keeps the result an upper
     bound (`online.py`).
5. **Decision.** The prediction is used, or the code runs and the point
   is learned (`emulator.py`).

## Benchmarks

The test case is ΛCDM on Planck 2018 lite + lowE + lensing, with
CosmoFit's MCMC running 4 chains in one process:

```bash
python benchmarks/chain.py --max-time 3600 --output chains/lcdm
python benchmarks/validate_chain.py chains/lcdm   # chain points re-checked with CAMB
```

One hour, starting with no training points, on an idle machine (CAMB
0.45 s per point):

| | v0.2 |
|---|---|
| evaluations | 131 203 |
| CAMB calls | 1 545 (897 learned, 648 checks) |
| emulated | 98.8% |
| equivalent with CAMB alone | 16.5 h |
| speed-up | 16.5× |
| `R - 1` after the hour | 0.36 |

Accuracy against CAMB, on 60 points drawn from the chain:

- `|Δ ln L|` has a median of 0.004 and a maximum of 0.02.
- The posterior agrees with Planck's: `H0` = 67.39 ± 0.51 and
  `n_s` = 0.9639 ± 0.0041.

Compared with v0.1 on the same machine:

- Evaluations in the first hour went from 54 000 to 131 000.
- A prediction takes 10 ms instead of 16 ms.
- A refit takes 4–7 s instead of 58 s.

The speed-up grows with the length of the run, because CAMB runs less
and less often once the emulator has learned the posterior.
