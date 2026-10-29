#!/usr/bin/env python3
"""Invert each sample's B-Sr-Nd isotopes for end-member mixing fractions.

    python run_mcmc.py example/endmembers.csv example/samples.csv --out out/

Writes <out>/summary.csv (median + 95 % interval + R-hat per sample and
end-member), <out>/<sample>_draws.npz (posterior draws) and, if matplotlib is
installed, <out>/<sample>_fractions.png.
"""
import argparse
import csv
import os

import numpy as np

from isomix import SYSTEMS, load_endmembers, load_samples, run_sample, summarise


def _plot(f, names, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    flat = f.reshape(-1, f.shape[-1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, n in enumerate(names):
        ax.hist(flat[:, i], bins=60, range=(0, 1), histtype="step", lw=1.5, label=n)
    ax.set_xlabel("mass fraction in source")
    ax.set_ylabel("posterior draws")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("endmembers")
    ap.add_argument("samples")
    ap.add_argument("--out", default="out")
    ap.add_argument("--systems", default=",".join(SYSTEMS),
                    help="comma list from %s (default: all)" % ",".join(SYSTEMS))
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="symmetric Dirichlet prior on fractions (1 = uniform)")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--tune", type=int, default=20000)
    ap.add_argument("--keep", type=int, default=20000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    systems = tuple(s.strip() for s in a.systems.split(","))
    names, mu, sd = load_endmembers(a.endmembers, systems)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name, obs, err in load_samples(a.samples, systems):
        res = run_sample(mu, sd, obs, err, alpha=np.full(len(names), a.alpha),
                         n_chains=a.chains, n_tune=a.tune, n_keep=a.keep,
                         thin=a.thin, seed=a.seed)
        summ = summarise(res["f"], names)
        print(f"\n{name}  (acceptance {res['accept'].mean():.2f})")
        for s in summ:
            flag = "" if s["rhat"] < 1.01 else "  <-- R-hat > 1.01, run longer"
            print(f"  {s['endmember']:<16} {s['median']:.3f}  "
                  f"[{s['lo95']:.3f}, {s['hi95']:.3f}]  R-hat {s['rhat']:.3f}{flag}")
            rows.append(dict(sample=name, **s))
        np.savez_compressed(os.path.join(a.out, f"{name}_draws.npz"),
                            f=res["f"], theta=res["theta"], endmembers=names,
                            systems=systems)
        _plot(res["f"], names, name, os.path.join(a.out, f"{name}_fractions.png"))

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {os.path.join(a.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
