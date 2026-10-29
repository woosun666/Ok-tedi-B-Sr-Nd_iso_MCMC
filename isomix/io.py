"""CSV readers. Column conventions are documented in README.md."""
import csv

import numpy as np

from .mixing import SYSTEMS


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def _num(s):
    s = (s or "").strip()
    return float(s) if s else np.nan


def load_endmembers(path, systems=tuple(SYSTEMS)):
    """-> names, mu (2S, K), sd (2S, K). Rows of mu/sd: [conc..., iso...]."""
    rows = _rows(path)
    names = [r["name"] for r in rows]
    cols = [SYSTEMS[s][0] for s in systems] + [SYSTEMS[s][1] for s in systems]
    mu = np.array([[_num(r[c]) for r in rows] for c in cols])
    sd = np.array([[_num(r[c + "_sd"]) for r in rows] for c in cols])
    if np.isnan(mu).any() or np.isnan(sd).any() or (sd <= 0).any():
        raise ValueError(f"{path}: every end-member needs a value and a "
                         f"positive _sd for {cols}")
    return names, mu, sd


def load_samples(path, systems=tuple(SYSTEMS)):
    """-> list of (name, obs (S,), err (S,)); blank isotope cells = NaN (skipped)."""
    out = []
    for r in _rows(path):
        iso = [SYSTEMS[s][1] for s in systems]
        obs = np.array([_num(r.get(c)) for c in iso])
        err = np.array([_num(r.get(c + "_err")) for c in iso])
        if (err[~np.isnan(obs)] <= 0).any() or np.isnan(err[~np.isnan(obs)]).any():
            raise ValueError(f"{path}: sample {r['name']} needs a positive _err "
                             "for every measured isotope value")
        out.append((r["name"], obs, err))
    return out
