"""Sampler checks. Run: python -m pytest tests/  (or python tests/test_isomix.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isomix import load_endmembers, mix, run_sample  # noqa: E402

EM = os.path.join(os.path.dirname(__file__), "..", "example", "endmembers.csv")


def test_mix_endmember_limits():
    _, mu, _ = load_endmembers(EM)
    for k in range(3):
        f = np.eye(3)[k]
        _, r = mix(f, mu[:3], mu[3:])
        assert np.allclose(r, mu[3:, k])


def test_no_data_returns_prior():
    # all observations missing -> posterior on f must be the Dirichlet(1)
    # prior: mean 1/3 each, var = 2/36 for K = 3
    _, mu, sd = load_endmembers(EM)
    res = run_sample(mu, sd, np.full(3, np.nan), np.ones(3), n_chains=4,
                     n_tune=10000, n_keep=40000, thin=10, seed=1)
    f = res["f"].reshape(-1, 3)
    assert np.allclose(f.mean(axis=0), 1 / 3, atol=0.03)
    assert np.allclose(f.var(axis=0), 2 / 36, atol=0.01)


def test_recovers_truth_with_tight_endmembers():
    _, mu, sd = load_endmembers(EM)
    sd = sd * 0.02                               # nearly fixed end-members
    truth = np.array([0.8, 0.12, 0.08])
    _, obs = mix(truth, mu[:3], mu[3:])
    err = np.array([0.5, 2e-5, 0.3])
    res = run_sample(mu, sd, obs, err, n_chains=4, n_tune=10000,
                     n_keep=20000, thin=10, seed=2)
    f = res["f"].reshape(-1, 3)
    lo, hi = np.percentile(f, [2.5, 97.5], axis=0)
    assert ((lo <= truth) & (truth <= hi)).all()
    assert np.allclose(np.median(f, axis=0), truth, atol=0.03)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
