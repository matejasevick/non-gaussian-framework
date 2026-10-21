"""Monte Carlo check of Lemma 1(c): Cov(Z_j, Z_l | beta) = alpha_j alpha_l across LD blocks,
and of the common-factor effect Cov(Z_j^2, Z_l^2) ~ Var(y^2) (problem P5)."""
import sys, numpy as np
sys.path.insert(0, '.')
from part3_martingale_checks import Design, sample_beta, B
rng = np.random.default_rng(7)
des = Design(20, rng)
beta = sample_beta(des, 0.5, 0.5, rng)
alpha = np.einsum('lab,lb->la', des.Sig, beta.reshape(des.L, B)).reshape(-1)
N, chunks = 400_000, 10
acc1 = np.zeros((des.p, des.p)); acc2 = np.zeros((des.p, des.p)); m2 = np.zeros(des.p); y2 = []
for _ in range(chunks):
    X = des.sample_X(N, rng); y = X @ beta + rng.normal(0, np.sqrt(0.5), N)
    Z = X * y[:, None] - alpha
    acc1 += Z.T @ Z; acc2 += (Z**2).T @ (Z**2); m2 += (Z**2).sum(0); y2.append(y**2)
tot = N * chunks
G = acc1 / tot; C2 = acc2 / tot - np.outer(m2 / tot, m2 / tot)
blk = np.repeat(np.arange(des.L), B); cross = blk[:, None] != blk[None, :]
err = (G - np.outer(alpha, alpha))[cross]
print("max |Cov(Z_j,Z_l) - alpha_j alpha_l| over cross-block pairs:", np.abs(err).max())
print("typical MC s.e. of an entry:", np.sqrt(np.median(np.outer(m2, m2)[cross]) / tot) / tot**0 / np.sqrt(tot) * np.sqrt(tot))
print("max |alpha_j alpha_l| (cross):", np.abs(np.outer(alpha, alpha)[cross]).max())
yy = np.concatenate(y2); print("Var(y^2) =", yy.var(), "; mean Cov(Z_j^2,Z_l^2) over cross pairs =", C2[cross].mean())
