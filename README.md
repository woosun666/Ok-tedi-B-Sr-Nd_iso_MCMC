# Ok-tedi-B-Sr-Nd_iso_MCMC

Bayesian source-mixing inversion of B–Sr–Nd isotopes (δ¹¹B, ⁸⁷Sr/⁸⁶Sr, εNd)
with MCMC, for the Ok Tedi porphyry Au(–Cu) system. Given K candidate
end-members (e.g. mantle wedge, slab fluid, sediment melt) with uncertain
concentrations and isotope values, it returns the posterior distribution of
each sample's mixing fractions.

> **Status: scaffold.** The model, sampler and tests work; the bundled
> end-member values and samples are **placeholders / synthetic** — replace them
> with your chosen literature end-members and the Ok Tedi data.

## Model

Concentration-weighted mixing, each isotope system with its own element:

```
C_mix = Σ fᵢ Cᵢ        R_mix = Σ fᵢ Cᵢ Rᵢ / C_mix
```

| Unknown | Prior |
|---|---|
| mixing fractions f (simplex) | Dirichlet(α), α = 1 → uniform (`--alpha`) |
| every end-member concentration and isotope value | Normal(mean, sd) from `endmembers.csv`, concentrations > 0 |

End-member values are sampled as nuisance parameters, so their uncertainty
propagates into f. Likelihood: independent Gaussian errors on each measured
isotope value; blank cells are skipped. Sampler: adaptive random-walk
Metropolis (numpy only), with multiple dispersed chains and split R-hat.

## Install & run

```bash
pip install -r requirements.txt
python run_mcmc.py example/endmembers.csv example/samples.csv --out out/
python tests/test_isomix.py
```

Options: `--systems B,Sr` (subset), `--alpha`, `--chains`, `--tune`, `--keep`,
`--thin`, `--seed`. Outputs: `out/summary.csv` (median, 95 % interval, R-hat
per sample × end-member), `out/<sample>_draws.npz`, `out/<sample>_fractions.png`.

## Input format

`endmembers.csv` — one row per end-member; every value needs a `_sd` column:

```
name,B_ppm,B_ppm_sd,d11B,d11B_sd,Sr_ppm,Sr_ppm_sd,Sr87_86,Sr87_86_sd,Nd_ppm,Nd_ppm_sd,eNd,eNd_sd
```

`samples.csv` — one row per sample; every measured value needs an `_err`
(1σ) column:

```
name,d11B,d11B_err,Sr87_86,Sr87_86_err,eNd,eNd_err
```

Lines starting with `#` are comments.

## Checks

`tests/test_isomix.py`:
- pure end-members reproduce their own isotope values;
- with no data the posterior on f equals the Dirichlet prior (sampler is unbiased);
- with near-fixed end-members a synthetic mixture is recovered within its 95 % interval.

## Known limitations / next steps

- **Convergence:** always check R-hat < 1.01 in the output. With broad
  end-member priors the posterior has strong f ↔ concentration trade-offs and
  long tails; the random-walk sampler then needs long runs (e.g.
  `--tune 40000 --keep 200000 --thin 50`). For production runs consider an
  HMC/NUTS sampler (PyMC, numpyro) on the same model.
- Each sample is inverted independently; a hierarchical version sharing
  end-member values across samples is the natural extension.
- Open-system processes (AFC, isotope fractionation during B degassing) are
  not modelled — mixing of source end-members only.
