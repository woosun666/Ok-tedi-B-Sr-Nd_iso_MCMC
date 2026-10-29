"""Bayesian mixing inversion with an adaptive Metropolis sampler (numpy only).

Unknowns for one sample
  * f      : K mixing fractions on the simplex, Dirichlet(alpha) prior.
             Sampled as K-1 unconstrained log-ratios z, f = softmax([0, z]);
             the Jacobian |df/dz| = prod(f) is folded into the prior.
  * theta  : every end-member concentration and isotope value, Normal(mu, sd)
             prior (nuisance parameters, so end-member uncertainty propagates
             into f). Sampled in standardised units u = (theta - mu) / sd.

Likelihood: independent Gaussian on each measured isotope value.
Sampler: Haario et al. (2001) adaptive Metropolis; covariance adapted during
the tuning phase only, then frozen so the kept draws form a valid chain.
"""
import numpy as np

from .mixing import mix


def _softmax0(z):
    a = np.concatenate(([0.0], z))
    a = np.exp(a - a.max())
    return a / a.sum()


def make_logpost(mu, sd, obs, err, alpha):
    n_sys = mu.shape[0] // 2
    k = mu.shape[1]
    meas = ~np.isnan(obs)

    def logpost(x):
        z, u = x[:k - 1], x[k - 1:].reshape(mu.shape)
        f = _softmax0(z)
        theta = mu + sd * u
        conc, iso = theta[:n_sys], theta[n_sys:]
        if (conc <= 0).any():
            return -np.inf
        _, r = mix(f, conc, iso)
        resid = (r[meas] - obs[meas]) / err[meas]
        return (np.sum(alpha * np.log(f))          # Dirichlet prior + Jacobian
                - 0.5 * np.sum(u * u)              # end-member priors
                - 0.5 * np.sum(resid * resid))     # likelihood
    return logpost


def _chain(logpost, x0, n_tune, n_keep, rng, thin, target=0.234):
    d = x0.size
    x, lp = x0.copy(), logpost(x0)
    chol = np.eye(d) * 0.1
    log_s = np.log(2.38 / np.sqrt(d))       # global step scale, tuned below
    hist = np.empty((n_tune, d))
    keep = np.empty((n_keep // thin, d))
    acc = 0
    for i in range(n_tune + n_keep):
        # tuning: shape from the latest half of the history (forgets the
        # transient), size by Robbins-Monro toward the target acceptance
        if i < n_tune and i >= 1000 and i % 200 == 0:
            cov = np.cov(hist[i // 2:i].T) + 1e-10 * np.eye(d)
            chol = np.linalg.cholesky(cov)
        xp = x + np.exp(log_s) * chol @ rng.standard_normal(d)
        lpp = logpost(xp)
        accepted = np.log(rng.random()) < lpp - lp
        if accepted:
            x, lp = xp, lpp
            if i >= n_tune:
                acc += 1
        if i < n_tune:
            log_s += (float(accepted) - target) / np.sqrt(i + 1) ** 0.6
        if i < n_tune:
            hist[i] = x
        elif (i - n_tune) % thin == 0 and (i - n_tune) // thin < keep.shape[0]:
            keep[(i - n_tune) // thin] = x
    return keep, acc / n_keep


def run_sample(mu, sd, obs, err, alpha=None, n_chains=4, n_tune=20000,
               n_keep=20000, thin=10, seed=0):
    """Return dict: f (chains, draws, K), theta (chains, draws, 2S, K), accept."""
    k = mu.shape[1]
    alpha = np.ones(k) if alpha is None else np.asarray(alpha, float)
    logpost = make_logpost(mu, sd, obs, err, alpha)
    rng = np.random.default_rng(seed)
    fs, thetas, accs = [], [], []
    for _ in range(n_chains):
        while True:                                   # dispersed prior start
            x0 = np.concatenate((rng.normal(0, 1, k - 1),
                                 rng.normal(0, 1, mu.size)))
            if np.isfinite(logpost(x0)):
                break
        draws, acc = _chain(logpost, x0, n_tune, n_keep, rng, thin)
        fs.append(np.array([_softmax0(d[:k - 1]) for d in draws]))
        thetas.append(mu + sd * draws[:, k - 1:].reshape(-1, *mu.shape))
        accs.append(acc)
    return {"f": np.array(fs), "theta": np.array(thetas), "accept": np.array(accs)}


def rhat(x):
    """Split-chain Gelman-Rubin R-hat for x of shape (chains, draws)."""
    n = x.shape[1] // 2
    s = np.concatenate((x[:, :n], x[:, n:2 * n]))
    w = s.var(axis=1, ddof=1).mean()
    b = n * s.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


def summarise(f, names):
    """Per end-member median, 2.5 / 97.5 % and R-hat of the mixing fraction."""
    flat = f.reshape(-1, f.shape[-1])
    q = np.percentile(flat, [2.5, 50, 97.5], axis=0)
    return [dict(endmember=n, median=q[1, i], lo95=q[0, i], hi95=q[2, i],
                 rhat=rhat(f[:, :, i])) for i, n in enumerate(names)]
