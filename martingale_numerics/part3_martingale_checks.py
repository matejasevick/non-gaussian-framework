#!/usr/bin/env python3
"""Numerical checks for the LD4M martingale section (part 3, non-Gaussian framework).

Model (Condition 4 of the section):
  * genotypes: blocks of B SNPs; each person carries two independent haplotypes; along a block an
    allele is copied from the previous SNP with probability rho_l and drawn fresh from Bern(f_l)
    otherwise.  Genotype = sum of the two haplotypes, standardised by the population mean and sd.
    Blocks are independent; Sigma_l is exactly AR(1) with parameter rho_l.
  * effects: spike-and-slab, beta_j = gamma_j * N(0, tau^2), gamma_j ~ Bern(pi), E beta_j^2 = h2/p.
  * noise: Gaussian, variance 1 - h2.

Experiments:
  A (beta fixed): exact Doob identity sum_i D_{L+i} = K - E[K|beta]; E[sum_i D_i^2 | beta] versus
    the Monte Carlo Var(K|beta); concentration of sum_i D_i^2 / Var(K|beta) (condition C1, sampling
    part); the same for weights supported on a single LD block (where C1 must fail).
  B (beta random): normality of (K - E K)/sd, share of Var(E[K|beta]) in Var(K), and the
    distribution-free lower bound on zeta^2 (Proposition 2).
  C (beta random): spread of zeta_beta^2 = Var(K|beta) across draws of beta (condition E1).
"""
import argparse
import itertools
import json
import math
import time

import numpy as np
from scipy import stats

B = 5  # SNPs per LD block


def geno_support(b, f, rho):
    """Support (standardised genotypes) and probabilities of one block."""
    pats = np.array(list(itertools.product([0, 1], repeat=b)), dtype=float)
    pf = np.where(pats == 1, f, 1 - f)
    prob = pf[:, 0].copy()
    for k in range(1, b):
        prob *= rho * (pats[:, k] == pats[:, k - 1]) + (1 - rho) * pf[:, k]
    G = (pats[:, None, :] + pats[None, :, :]).reshape(-1, b)
    P = (prob[:, None] * prob[None, :]).reshape(-1)
    return (G - 2 * f) / math.sqrt(2 * f * (1 - f)), P


class Design:
    def __init__(self, p, rng):
        self.L = p // B
        self.p = self.L * B
        self.f = rng.uniform(0.1, 0.5, self.L)
        self.rho = rng.uniform(0.3, 0.95, self.L)
        sup = [geno_support(B, f, r) for f, r in zip(self.f, self.rho)]
        self.Xs = np.stack([s[0] for s in sup])          # L x S x B
        self.Ps = np.stack([s[1] for s in sup])          # L x S
        lag = np.abs(np.subtract.outer(np.arange(B), np.arange(B)))
        self.Sig = np.stack([r ** lag for r in self.rho])
        S_enum = np.einsum('ls,lsa,lsb->lab', self.Ps, self.Xs, self.Xs)
        assert np.allclose(S_enum, self.Sig, atol=1e-10), "Sigma is not AR(1)"
        u = (self.Sig ** 4).sum(axis=2).reshape(-1)
        self.z = u - u.mean()
        self.c = self.z / (self.z @ self.z)
        # weights carried by one LD block only (C1 must fail for these)
        zb = self.z[:B] - self.z[:B].mean()
        self.c1 = np.zeros(self.p)
        self.c1[:B] = zb / (zb @ zb)
        # z' Sigma^{o4} z (block diagonal)
        zl = self.z.reshape(self.L, B)
        self.zS4z = np.einsum('la,lab,lb->', zl, self.Sig ** 4, zl)

    def sample_X(self, n, rng):
        L, f, rho = self.L, self.f[None, :, None], self.rho[None, :, None]

        def hap():
            U = rng.random((n, L, B)) < f
            C = rng.random((n, L, B)) < rho
            h = np.empty((n, L, B), dtype=np.int8)
            h[:, :, 0] = U[:, :, 0]
            for k in range(1, B):
                h[:, :, k] = np.where(C[:, :, k], h[:, :, k - 1], U[:, :, k])
            return h

        g = hap() + hap()
        return ((g - 2 * f) / np.sqrt(2 * f * (1 - f))).reshape(n, -1)


def sample_beta(des, h2, pi, rng):
    tau = math.sqrt(h2 / (des.p * pi))
    return np.where(rng.random(des.p) < pi, rng.normal(0.0, tau, des.p), 0.0)


def cond_moments(des, beta, s2):
    """alpha = Sigma beta and mu[:, a] = E[(x_j y - alpha_j)^a | beta], a = 2, 3, 4 (exact)."""
    L = des.L
    bB = beta.reshape(L, B)
    alpha = np.einsum('lab,lb->la', des.Sig, bB)                       # L x B
    d = np.einsum('lsb,lb->ls', des.Xs, bB)                            # delta_B on the support
    m2, m3, m4 = [(des.Ps * d ** k).sum(axis=1) for k in (2, 3, 4)]
    kap = np.stack([m2, m3, m4 - 3 * m2 ** 2], axis=1)                 # cumulants 2..4 of delta_B
    T = kap.sum(axis=0)
    k2 = T[0] - kap[:, 0] + s2                                         # cumulants of eta = y - delta_B
    k3 = T[1] - kap[:, 1]
    k4 = T[2] - kap[:, 2]
    em = np.stack([np.ones(L), np.zeros(L), k2, k3, k4 + 3 * k2 ** 2], axis=1)  # moments of eta
    W = des.Xs * d[:, :, None] - alpha[:, None, :]                     # x_j delta_B - alpha_j
    mu = np.zeros((L, B, 5))
    for a in (2, 3, 4):
        for t in range(a + 1):
            mu[:, :, a] += math.comb(a, t) * np.einsum(
                'ls,lsb->lb', des.Ps, W ** (a - t) * des.Xs ** t) * em[:, t][:, None]
    return alpha.reshape(-1), mu.reshape(-1, 5)


def m4(s, r, mu):
    return (s ** 4 + 6 * r * mu[:, 2] * s ** 2 + 4 * r * mu[:, 3] * s + r * mu[:, 4]
            + 3 * r * (r - 1) * mu[:, 2] ** 2)


def path(des, beta, alpha, mu, n, s2, rng, weights):
    """One sample of rows given beta: K for each weight vector and the row increments D_{L+i}."""
    X = des.sample_X(n, rng)
    y = X @ beta + rng.normal(0.0, math.sqrt(s2), n)
    Z = X * y[:, None] - alpha[None, :]
    S = n * alpha[None, :] + np.cumsum(Z, axis=0)
    Sp = np.vstack([n * alpha[None, :], S[:-1]])
    r = (n - np.arange(1, n + 1))[:, None].astype(float)
    mu2, mu3, mu4 = mu[:, 2], mu[:, 3], mu[:, 4]
    psi = (4 * Z * (Sp ** 3 + 3 * r * mu2 * Sp + r * mu3) + 6 * (Z ** 2 - mu2) * (Sp ** 2 + r * mu2)
           + 4 * (Z ** 3 - mu3) * Sp + (Z ** 4 - mu4))
    ahat4 = (S[-1] / n) ** 4
    return [(w @ ahat4, psi @ w / n ** 4) for w in weights]


def moments_summary(x):
    t = (x - x.mean()) / x.std(ddof=1)
    return dict(skew=float(stats.skew(x)), exkurt=float(stats.kurtosis(x)),
                ks=float(stats.kstest(t, 'norm').statistic), p196=float(np.mean(np.abs(t) > 1.96)))


def exp_A(p, reps, seed, h2=0.5, pi=0.1):
    rng = np.random.default_rng(seed)
    des = Design(p, rng)
    n, s2 = des.p, 1 - h2
    beta = sample_beta(des, h2, pi, rng)
    alpha, mu = cond_moments(des, beta, s2)
    out = {}
    res = {k: dict(K=[], SD=[], SD2=[]) for k in ('z', 'one_block')}
    for _ in range(reps):
        for key, (K, D) in zip(res, path(des, beta, alpha, mu, n, s2, rng, [des.c, des.c1])):
            res[key]['K'].append(K)
            res[key]['SD'].append(D.sum())
            res[key]['SD2'].append((D ** 2).sum())
    for key, w in zip(res, (des.c, des.c1)):
        K, SD, SD2 = (np.array(res[key][v]) for v in ('K', 'SD', 'SD2'))
        EK = (w * m4(n * alpha, n, mu)).sum() / n ** 4
        vK = K.var(ddof=1)
        out[key] = dict(
            identity_max_rel_err=float(np.max(np.abs(SD - (K - EK))) / np.sqrt(vK)),
            mean_K_minus_EK_over_sd=float((K.mean() - EK) / np.sqrt(vK / reps)),
            ratio_E_sumD2_to_varK=float(SD2.mean() / vK),
            cv_sumD2=float(SD2.std(ddof=1) / SD2.mean()),
            **moments_summary(K))
    return dict(exp='A', p=p, n=n, reps=reps, h2=h2, pi=pi, L=des.L, **out)


def exp_B(p, reps, seed, h2=0.5, pi=0.1):
    rng = np.random.default_rng(seed)
    des = Design(p, rng)
    n, s2 = des.p, 1 - h2
    K, EKb = np.empty(reps), np.empty(reps)
    for t in range(reps):
        beta = sample_beta(des, h2, pi, rng)
        alpha, mu = cond_moments(des, beta, s2)
        X = des.sample_X(n, rng)
        y = X @ beta + rng.normal(0.0, math.sqrt(s2), n)
        K[t] = des.c @ (X.T @ y / n) ** 4
        EKb[t] = (des.c * m4(n * alpha, n, mu)).sum() / n ** 4
    vK = K.var(ddof=1)
    lb = 24 * s2 ** 4 * n * (n - 1) * (n - 2) * (n - 3) / n ** 8 * des.zS4z / (des.z @ des.z) ** 2
    return dict(exp='B', p=p, n=n, reps=reps, h2=h2, pi=pi, L=des.L,
                mean_K=float(K.mean()), sd_K=float(np.sqrt(vK)),
                share_var_beta_part=float(EKb.var(ddof=1) / vK),
                n4_z2_varK=float(n ** 4 * (des.z @ des.z) * vK),
                lower_bound_ratio=float(vK / lb), **moments_summary(K))


def exp_C(p, reps, seed, h2=0.5, pi=0.1, T=10):
    """Spread of zeta_beta^2 = Var(K | beta) across draws of beta (condition E1).

    For each of T draws of beta, zeta_beta^2 is estimated by the Monte Carlo mean of sum_i D_i^2,
    which is unbiased by the martingale identity E[sum_i D_i^2 | beta] = Var(K | beta).
    """
    rng = np.random.default_rng(seed)
    des = Design(p, rng)
    n, s2 = des.p, 1 - h2
    zeta2, se, kap = [], [], []
    for _ in range(T):
        beta = sample_beta(des, h2, pi, rng)
        alpha, mu = cond_moments(des, beta, s2)
        sd2 = np.array([(path(des, beta, alpha, mu, n, s2, rng, [des.c])[0][1] ** 2).sum()
                        for _ in range(reps)])
        zeta2.append(sd2.mean())
        se.append(sd2.std(ddof=1) / math.sqrt(reps))
        kap.append((des.c * m4(n * alpha, n, mu)).sum() / n ** 4)
    zeta2, se, kap = map(np.array, (zeta2, se, kap))
    var_between = max(0.0, zeta2.var(ddof=1) - np.mean(se ** 2))
    return dict(exp='C', p=p, n=n, reps=reps, T=T, h2=h2, pi=pi, L=des.L,
                mean_zeta2=float(zeta2.mean()), cv_zeta2_raw=float(zeta2.std(ddof=1) / zeta2.mean()),
                cv_zeta2_mc_corrected=float(math.sqrt(var_between) / zeta2.mean()),
                mean_mc_se_rel=float(np.mean(se) / zeta2.mean()),
                share_var_kappa=float(kap.var(ddof=1) / (zeta2.mean() + kap.var(ddof=1))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--p', type=int, required=True)
    ap.add_argument('--reps', type=int, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--pi', type=float, default=0.1)
    a = ap.parse_args()
    t0 = time.time()
    kw = dict(pi=a.pi, T=a.T) if a.exp == 'C' else dict(pi=a.pi)
    res = {'A': exp_A, 'B': exp_B, 'C': exp_C}[a.exp](a.p, a.reps, a.seed, **kw)
    res['seconds'] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)
