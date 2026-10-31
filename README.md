# fastcmb

fastcmb is an online emulator for Boltzmann codes, aimed at CMB analyses.
It sits in front of CAMB, or any other code, and works one point at a
time:

- It predicts the spectra together with the error of that prediction,
  measured in units of `ln L`.
- If the predicted error is below the tolerance, it returns the
  prediction. This takes about 6 ms.
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
pip install "fastcmb[camb]"      # with CAMB as the oracle
pip install "fastcmb[cobaya]"    # and the cobaya theory
pip install "cosmofit[fast]"     # CosmoFit, with fastcmb and CAMB
```

The emulator alone (`pip install fastcmb`) needs only numpy and scipy;
the oracle is then a code of your own. From a clone, `pip install -e ".[dev]"`
installs the tests' dependencies as well.

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
`CAMBOracle` takes the parameters as cobaya names them (`H0` or
`cosmomc_theta`, `ombh2`, `omch2`, `ln1e10As`/`logA` or `As`, `ns`, `tau`,
and optionally `mnu`, `nnu`, `omk`, `w`, `wa`).

`emulator(params, loose_tolerance=1.0)` also answers with a prediction
whose error is above the tolerance but below 1, marked
`result.approximate`. A sampler can use it to reject a proposal that is
clearly worse than the current point without running the code, and ask
again without `loose_tolerance` for one it might accept. CosmoFit's MCMC
does this with `screen: true`.

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

The theory takes every `camb` option and parameter, and dark sectors
beyond ΛCDM: CPL and other dark energies, early dark energy, and
Hu–Sawicki f(R) when CAMB is MGCAMB. Their parameters become inputs of
the emulator, together with the distance to last scattering, along which
they trade off against `h`.

## Options

| | default | |
|---|---|---|
| `tolerance` | 0.05 | largest predicted `ln L` error the emulator answers with |
| `sigma8_tolerance` | 1e-3 | largest predicted relative error of `sigma8` |
| `min_train` | 30 | points computed by the code before the first fit |
| `retrain_fraction` | 0.1 | refit once the new points are this fraction of those fitted on |
| `max_train` | 800 | most points fitted on (the latest) |
| `training_file` | none | keeps the points, so parallel chains and later runs share them |
| `check_every` | 200 | after N predictions the code runs as well, to measure the error; the interval doubles while checks pass, up to 10 N |

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
- A prediction takes 6 ms instead of 16 ms.
- A refit takes 4–7 s instead of 58 s.

The speed-up grows with the length of the run, because CAMB runs less
and less often once the emulator has learned the posterior.

Starting from a trained emulator (the hour above), 20 minutes:

| | 1 process | 4 processes (`processes: 4`) |
|---|---|---|
| MCMC steps | 10 080 | 26 880 |
| CAMB calls (first process) | 93 | 40 |
| `R - 1` | 13.9 | **0.026** |
| `\|Δ ln L\|` vs CAMB, max | 0.016 | 0.011 |

Several processes share their points through `training_file`, so give
it whenever a run has more than one.

### Beyond ΛCDM

The same chain with a dark sector, through CosmoFit (`--dark-energy`
`cpl`, `ede` or `hu_sawicki`; `--bao` adds DESI BAO), 4 processes sharing
a training file, run until `R - 1 < 0.02`:

| | CPL + DESI BAO | EDE | Hu–Sawicki f(R) |
|---|---|---|---|
| sampled beyond ΛCDM | `w0`, `wa` | `f_ede`, `log10 z_c` | `log10 \|f_R0\|` |
| wall time | 33 min | 39 min | 59 min |
| `R - 1` | 0.019 | 0.019 | 0.016 |
| code calls / evaluations, per process | 133 / 18 044 | 426 / 20 201 | 157 / 5 688 |
| `\|Δ ln L\|` vs the code, median / max | — | 0.015 / 0.10 | 0.003 / 0.032 |
| speed-up | — | 8.6× | 8.3× |

f(R) runs through MGCAMB, roughly ten times slower than CAMB per point.
The CPL run was not re-checked against CAMB point by point; the analysis
below does that for CPL with more data.

## Analyses

[`analysis/cpl_vs_lcdm/`](https://github.com/salihyesil59/fastcmb/tree/main/analysis/cpl_vs_lcdm) fits ΛCDM and CPL
(`w0`, `wa`) with CosmoFit to the full Planck 2018 likelihoods (plik_lite,
lowE, lensing), DESI DR2 BAO, cosmic chronometers and, in turn, Pantheon+,
Union3 and DES-SN5YR: six runs to `R - 1 < 0.01`, all six ΛCDM parameters
sampled. The notebook compares the models (Δχ², AIC, BIC), shows which data
carry the signal, and checks the emulator against CAMB.

| | Pantheon+ | Union3 | DES-SN5YR (Dovekie) |
|---|---|---|---|
| `w0` | −0.843 ± 0.054 | −0.677 ± 0.090 | −0.809 ± 0.056 |
| `wa` | −0.59 ± 0.20 | −1.04 ± 0.30 | −0.70 ± 0.22 |
| preference for CPL | 2.6σ | 3.5σ | 3.1σ |

Each run took 15–52 min. CAMB ran at 272 (ΛCDM) and 642 (CPL) points
over all six runs, and `|Δ ln L|` against CAMB is at most 0.015.
