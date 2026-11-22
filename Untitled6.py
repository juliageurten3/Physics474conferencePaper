#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NGC 1052-DF2: dynamical mass from ten globular clusters
# Follows van Dokkum et al. (2018), Nature 555, 629.
# Cells are separated by "# %%" so this runs cell-by-cell in Jupyter.

# %% 0. Imports, constants, data
import math
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---- constants given by you ----
G = 4.30091e-6        # kpc (km/s)^2 / Msun
MSTAR = 2.0e8         # stellar mass, Msun
R_OUT = 7.6           # radius of the outermost cluster, kpc
R_HALF = 3.1          # half-number radius of the cluster system, kpc
SIGMA_BASE = 10.5     # published 90% upper limit on the intrinsic dispersion, km/s
V_MEAN = 1803.0       # mean velocity of the clusters, km/s

# ---- baseline dynamical assumptions (from the paper) ----
ALPHA0 = 0.0          # potential slope: isothermal (flat rotation curve)
BETA0 = 0.0           # orbit anisotropy: isotropic orbits
GAMMA0 = 0.9          # slope of the cluster density profile (paper: 0.9 +/- 0.3)

# ---- ranges for the "vary everything together" Monte Carlo ----
# Taken from the values you listed for the one-at-a-time tests.
ALPHA_RANGE = (-1.0, 1.0)
BETA_RANGE = (-1.0, 0.75)
GAMMA_RANGE = (0.5, 1.5)

# ---- numerical settings ----
SEED = 1
N_FORWARD = 1000      # fake data sets per forward-model test (paper uses 1,000)
N_BOOT = 500          # bootstrap resamples (you asked for 500)
N_MC = 10000          # random draws in the all-assumptions Monte Carlo (my choice)
BIWEIGHT_C = 9.0      # tuning constant of the biweight (standard value, Beers+1990)
SIG_GRID_MAX = 100.0  # largest dispersion considered in the likelihood, km/s
N_GRID = 20001        # number of grid points for the likelihood

# ---- the ten clusters (authors' table) ----
ids = ["GC-39", "GC-59", "GC-71", "GC-73", "GC-77",
       "GC-85", "GC-91", "GC-92", "GC-98", "GC-101"]
r = np.array([7.55, 4.91, 2.57, 6.77, 0.40, 2.26, 1.55, 1.94, 3.59, 4.77])  # kpc
cz = np.array([1818, 1799, 1805, 1814, 1804, 1801, 1802, 1789, 1764, 1800], float)
ev_min = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14], float)
ev_plus = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13], float)

dv = cz - V_MEAN                  # velocity relative to the mean
err = (ev_min + ev_plus) / 2.0    # symmetric measurement error
N = len(dv)

# ---- published numbers from the paper, used only for comparison ----
PUB_M_TME = 3.4e8            # Msun, mass within R_OUT
PUB_M_HALF = 3.2e8           # Msun, mass within R_HALF
PUB_SIG_NMAD = 4.7           # observed spreads, km/s
PUB_SIG_BI = 8.4
PUB_SIG_RMS = 14.3
PUB_INTR = (3.2, 5.5, 3.2)   # biweight intrinsic dispersion: best, +, -
PUB_INTR_RMS = 12.2          # intrinsic dispersion if the rms is used
PUB_M_RMS = 5e8              # mass if the rms is used
# mass ratios (changed assumption / baseline) quoted in the paper
PUB_RATIO = {"alpha=+1": 0.90, "alpha=-1": 0.80,
             "beta=-1": 4.2 / 3.4, "beta=0.5": 2.4 / 3.4,
             "gamma=0.5": 0.80, "gamma=1.5": 1.30}

summary = []   # (label, mass) pairs, filled as we go, used in the final cell


# %% 1. Helper functions
def C_factor(alpha, beta, gamma):
    """Constant C in the tracer mass estimator (paper, Eq. 4).
    It depends on the potential slope alpha, orbit anisotropy beta
    and cluster density slope gamma."""
    gam = 4.0 * math.gamma(alpha / 2 + 2.5) / (math.sqrt(math.pi) * math.gamma(alpha / 2 + 1))
    return gam * (alpha + gamma + 1 - 2 * beta) / (alpha + 3 - beta * (alpha + 2))


def biweight_scale(x, c=BIWEIGHT_C):
    """Biweight estimate of the spread (robust against outliers).
    Works on a 2D array: one spread per row."""
    x = np.atleast_2d(x)
    med = np.median(x, axis=1, keepdims=True)
    d = x - med
    mad = np.median(np.abs(d), axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = d / (c * mad)
        keep = np.abs(u) < 1
        num = np.sum(np.where(keep, d**2 * (1 - u**2) ** 4, 0.0), axis=1)
        den = np.abs(np.sum(np.where(keep, (1 - u**2) * (1 - 5 * u**2), 0.0), axis=1))
        out = np.sqrt(x.shape[1] * num) / den
    out[mad[:, 0] == 0] = np.nan      # spread undefined if MAD is zero
    return out


def nmad_scale(x):
    """Normalised median absolute deviation (1.4826 makes it match sigma for a Gaussian)."""
    x = np.atleast_2d(x)
    med = np.median(x, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=1)


def rms_scale(x):
    """Root-mean-square of the velocities about zero (zero = V_MEAN)."""
    x = np.atleast_2d(x)
    return np.sqrt(np.mean(x**2, axis=1))


def one_value(stat, x):
    """Apply a spread estimator to a single 1D data set and return a number."""
    return float(stat(np.atleast_2d(x))[0])


def tme_mass(sigma_intr, alpha=ALPHA0, beta=BETA0, gamma=GAMMA0, dv_=dv, r_=r):
    """Tracer mass estimator (paper, Eq. 3): mass inside R_OUT.
    Velocities are rescaled so their spread equals the intrinsic dispersion,
    then the biweight spread of v * r^(alpha/2) is used.
    For alpha = 0 this reduces to M = (C/G) sigma^2 R_OUT."""
    C = C_factor(alpha, beta, gamma)
    scale = sigma_intr / one_value(biweight_scale, dv_)   # f^-1 = sigma_intr / sigma_obs
    s = one_value(biweight_scale, dv_ * scale * r_ ** (alpha / 2))
    return C / G * s**2 * R_OUT ** (1 - alpha)


def half_number_mass(sigma_intr):
    """Mass inside the half-number radius, M = 4 sigma^2 R / G (Wolf et al. 2010, ref. 17 of the paper)."""
    return 4.0 * sigma_intr**2 * R_HALF / G


# Random numbers for the forward model. They are drawn ONCE and reused, so the
# result changes smoothly with sigma and brentq can find a clean root.
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_FORWARD, N))   # intrinsic scatter, in units of sigma
E = rng.standard_normal((N_FORWARD, N))   # measurement noise, in units of the errors


def sigma_at_percentile(obs, stat, pct, errs):
    """Forward-model Monte Carlo.
    For a trial intrinsic dispersion s, build fake clusters = s*Z + errs*E and measure
    their spread with `stat`. Find the s where `obs` sits at the `pct`-th percentile
    of the fake spreads:  pct=50 -> best fit, 10 -> 90% upper limit,
    16 / 84 -> the +1 sigma / -1 sigma ends."""
    n = len(errs)
    z, e = Z[:, :n], E[:, :n]

    def f(s):
        return np.percentile(stat(s * z + errs * e), pct) - obs

    if f(0.0) >= 0:          # noise alone already gives a spread >= observed
        return 0.0
    hi = 10 * obs + 100.0
    return brentq(f, 0.0, hi)


def lnlike(dv_, err_, sig):
    """Gaussian log-likelihood for a set of sigma values (mean fixed at V_MEAN).
    Each cluster has total variance = sigma^2 + its own error^2. All clusters are kept."""
    s2 = np.atleast_1d(sig)[:, None] ** 2 + err_[None, :] ** 2
    return -0.5 * np.sum(np.log(2 * np.pi * s2) + dv_[None, :] ** 2 / s2, axis=1)


def likelihood_best_and_limit(dv_, err_):
    """Maximum-likelihood sigma and 90% upper limit (flat prior on sigma >= 0)."""
    # best fit: where d(lnL)/d(sigma) = 0, i.e. sum((dv^2 - s2)/s2^2) = 0
    def slope(sig):
        s2 = sig**2 + err_**2
        return np.sum((dv_**2 - s2) / s2**2)

    if slope(0.0) <= 0:
        best = 0.0
    else:
        best = brentq(slope, 0.0, 2 * np.max(np.abs(dv_)) + np.max(err_))

    # 90% limit: the sigma below which 90% of the (normalised) likelihood lies
    grid = np.linspace(0.0, SIG_GRID_MAX, N_GRID)
    ll = lnlike(dv_, err_, grid)
    L = np.exp(ll - ll.max())
    cdf = np.cumsum(L) / np.sum(L)
    limit = brentq(lambda s: np.interp(s, grid, cdf) - 0.9, 0.0, SIG_GRID_MAX)
    return best, limit


def show_mass(label, M, store=True):
    """Print a mass with M/M* and M - M*  (does the stellar mass alone explain it?)."""
    print(f"{label:<42s} M = {M:9.3e}   M/M* = {M / MSTAR:6.2f}   M-M* = {M - MSTAR:+10.2e}")
    if store:
        summary.append((label, M))


def compare(label, mine, pub, unit=""):
    print(f"{label:<44s} mine = {mine:9.3g}   published = {pub:9.3g} {unit}  ({100 * (mine - pub) / pub:+.0f}%)")


# %% 2. FIRST: reproduce the published results
print("=== 1. Reproduce the paper ===\n")

M_base = tme_mass(SIGMA_BASE)
M_half = half_number_mass(SIGMA_BASE)
compare("Mass inside R_OUT (sigma = 10.5)", M_base, PUB_M_TME, "Msun")
compare("Mass inside R_HALF (sigma = 10.5)", M_half, PUB_M_HALF, "Msun")
show_mass("Baseline TME (R_OUT)", M_base)
show_mass("Half-number radius", M_half)

print("\nObserved spreads (km/s):")
obs_bi = one_value(biweight_scale, dv)
obs_nmad = one_value(nmad_scale, dv)
obs_rms = one_value(rms_scale, dv)
compare("biweight", obs_bi, PUB_SIG_BI, "km/s")
compare("NMAD", obs_nmad, PUB_SIG_NMAD, "km/s")
compare("rms", obs_rms, PUB_SIG_RMS, "km/s")

print("\nIntrinsic dispersion with the biweight (km/s):")
best = sigma_at_percentile(obs_bi, biweight_scale, 50, err)
lo = sigma_at_percentile(obs_bi, biweight_scale, 84, err)
hi = sigma_at_percentile(obs_bi, biweight_scale, 16, err)
lim = sigma_at_percentile(obs_bi, biweight_scale, 10, err)
print(f"  mine      : {best:.1f} +{hi - best:.1f} -{best - lo:.1f},  90% limit {lim:.1f}")
print(f"  published : {PUB_INTR[0]} +{PUB_INTR[1]} -{PUB_INTR[2]},  90% limit {SIGMA_BASE}")
best_rms = sigma_at_percentile(obs_rms, rms_scale, 50, err)
compare("intrinsic dispersion if rms is used", best_rms, PUB_INTR_RMS, "km/s")

print("\nSensitivity ratios, mass / baseline mass (sigma = 10.5):")
cases = {"alpha=+1": dict(alpha=1.0), "alpha=-1": dict(alpha=-1.0),
         "beta=-1": dict(beta=-1.0), "beta=0.5": dict(beta=0.5),
         "gamma=0.5": dict(gamma=0.5), "gamma=1.5": dict(gamma=1.5)}
for name, kw in cases.items():
    ratio = tme_mass(SIGMA_BASE, **kw) / M_base
    compare(name, ratio, PUB_RATIO[name])


# %% 3. SECOND: change the dispersion estimator
print("\n=== 2. Different dispersion estimators (baseline dynamics) ===\n")
estimators = {"biweight": biweight_scale, "NMAD": nmad_scale, "rms": rms_scale}
for name, stat in estimators.items():
    obs = one_value(stat, dv)
    s_best = sigma_at_percentile(obs, stat, 50, err)
    s_90 = sigma_at_percentile(obs, stat, 10, err)
    print(f"{name}: observed {obs:.1f}, intrinsic best {s_best:.1f}, 90% limit {s_90:.1f} km/s")
    show_mass(f"  {name} best fit", tme_mass(s_best))
    show_mass(f"  {name} 90% limit", tme_mass(s_90))

s_best, s_90 = likelihood_best_and_limit(dv, err)
print(f"Gaussian likelihood (all ten): best {s_best:.1f}, 90% limit {s_90:.1f} km/s")
show_mass("  likelihood best fit", tme_mass(s_best))
show_mass("  likelihood 90% limit", tme_mass(s_90))
compare("rms 90%-limit... mass vs paper's rms mass", tme_mass(sigma_at_percentile(obs_rms, rms_scale, 10, err)), PUB_M_RMS, "Msun")


# %% 4. THIRD: change the dynamical assumptions, one at a time
print("\n=== 3. One assumption at a time (sigma = 10.5) ===\n")
cases3 = {"baseline": {}, "alpha=-1": dict(alpha=-1.0), "alpha=+1": dict(alpha=1.0),
          "beta=-1": dict(beta=-1.0), "beta=0.5": dict(beta=0.5), "beta=0.75": dict(beta=0.75),
          "gamma=0.5": dict(gamma=0.5), "gamma=1.5": dict(gamma=1.5)}
for name, kw in cases3.items():
    M = tme_mass(SIGMA_BASE, **kw)
    show_mass(f"{name} (x{M / M_base:.2f} baseline)", M, store=(name != "baseline"))


# %% 5. FOURTH: vary all assumptions together
print("\n=== 4. Monte Carlo over alpha, beta, gamma (sigma = 10.5) ===\n")
rng_mc = np.random.default_rng(SEED + 1)
a_draw = rng_mc.uniform(*ALPHA_RANGE, N_MC)   # uniform draws inside each range
b_draw = rng_mc.uniform(*BETA_RANGE, N_MC)
g_draw = rng_mc.uniform(*GAMMA_RANGE, N_MC)

masses_mc = []
for a, b, g in zip(a_draw, b_draw, g_draw):
    if C_factor(a, b, g) <= 0:     # negative C gives a negative mass: not physical, skip
        continue
    masses_mc.append(tme_mass(SIGMA_BASE, a, b, g))
masses_mc = np.array(masses_mc)
print(f"Draws kept: {len(masses_mc)} of {N_MC} ({N_MC - len(masses_mc)} dropped because C <= 0)")
print(f"Mean {masses_mc.mean():.2e}, standard deviation {masses_mc.std():.2e} Msun")
for p in (5, 16, 50, 84, 95):
    show_mass(f"MC percentile {p}", np.percentile(masses_mc, p), store=(p in (16, 50, 84)))


# %% 6. FIFTH: change the sample (leave-one-out, then bootstrap)
print("\n=== 5a. Leave-one-out (biweight, 90% limit) ===\n")
loo_masses = []
for i in range(N):
    keep = np.arange(N) != i
    dv_i, err_i, r_i = dv[keep], err[keep], r[keep]
    obs_i = one_value(biweight_scale, dv_i)
    s90_i = sigma_at_percentile(obs_i, biweight_scale, 10, err_i)
    M_i = tme_mass(s90_i, dv_=dv_i, r_=r_i)
    loo_masses.append(M_i)
    print(f"drop {ids[i]:<7s} observed {obs_i:5.1f}  90% limit {s90_i:5.1f} km/s", end="   ")
    show_mass(f"LOO without {ids[i]}", M_i, store=False)
loo_masses = np.array(loo_masses)
print(f"Leave-one-out mass range: {loo_masses.min():.2e} to {loo_masses.max():.2e} Msun")

print(f"\n=== 5b. Bootstrap, {N_BOOT} resamples (biweight, 90% limit) ===\n")
rng_b = np.random.default_rng(SEED + 2)
boot_masses, skipped = [], 0
for _ in range(N_BOOT):
    idx = rng_b.integers(0, N, N)          # pick 10 clusters with replacement
    dv_b, err_b, r_b = dv[idx], err[idx], r[idx]
    obs_b = one_value(biweight_scale, dv_b)
    if not np.isfinite(obs_b):             # too many repeats: biweight undefined
        skipped += 1
        continue
    s90_b = sigma_at_percentile(obs_b, biweight_scale, 10, err_b)
    boot_masses.append(tme_mass(s90_b, dv_=dv_b, r_=r_b))
boot_masses = np.array(boot_masses)
print(f"Resamples used: {len(boot_masses)} ({skipped} skipped, biweight undefined)")
for p in (5, 16, 50, 84, 95):
    show_mass(f"Bootstrap percentile {p}", np.percentile(boot_masses, p), store=(p in (16, 50, 84)))
print(f"Fraction of resamples with M <= M*: {np.mean(boot_masses <= MSTAR):.2f}")


# %% 7. LASTLY: compare every result with the stellar mass
print("\n=== 6. All results vs the stellar mass (M* = 2e8 Msun) ===\n")
print(f"{'result':<42s} {'M (Msun)':>10s} {'M/M*':>7s} {'M-M*':>11s}   verdict")
for label, M in summary:
    if M <= MSTAR:
        verdict = "stars alone can explain it"
    else:
        verdict = "room for extra mass (upper limit)"
    print(f"{label:<42s} {M:10.2e} {M / MSTAR:7.2f} {M - MSTAR:+11.2e}   {verdict}")

# Two histograms: how much the mass moves with the assumptions and with the sample
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in [(axes[0], masses_mc, "All assumptions varied (MC)"),
                        (axes[1], boot_masses, "Bootstrap of the ten clusters")]:
    ax.hist(data / 1e8, bins=40, color="0.7")
    ax.axvline(MSTAR / 1e8, color="C1", label="stellar mass")
    ax.axvline(PUB_M_TME / 1e8, color="C0", ls="--", label="published 3.4e8")
    ax.set_xlabel("Mass within 7.6 kpc  [1e8 Msun]")
    ax.set_title(title)
axes[0].set_ylabel("Number")
axes[0].legend()
plt.tight_layout()
plt.show()


# In[ ]:




