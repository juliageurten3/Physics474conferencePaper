#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
df2_analysis.py  -  mass of NGC 1052-DF2 from its globular cluster velocities

Everything is in this one file. Each "# %%" line starts a new notebook cell, so you can
paste the file into Jupyter / Colab cell by cell, or just run it from top to bottom.

Method: the tracer mass estimator (TME) used by van Dokkum et al. (2018, Nature 555, 629).
Data: the authors' public table, http://www.astro.yale.edu/dokkum/outgoing/ascii_table.txt

AI assistance note: (fill this in as your course policy requires, and keep it in the file)
"""

# %% 0. Setup ---------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from math import gamma as gamma_fn, pi, sqrt
from scipy.optimize import brentq

G = 4.30091e-6        # gravitational constant in kpc (km/s)^2 / Msun
MSTAR = 2.0e8         # stellar mass of DF2 in Msun (van Dokkum et al. 2018)
R_OUT = 7.6           # kpc, radius of the outermost cluster (published)
R_HALF = 3.1          # kpc, half-number radius of the cluster system (published)
SIGMA_BASE = 10.5     # km/s, published 90% upper limit on the intrinsic dispersion

os.makedirs("figs", exist_ok=True)

# %% 1. Data ----------------------------------------------------------------------------
# One row per cluster: name, radial velocity cz (km/s), error below, error above, radius (kpc)
# (columns r_kpc, v, ev_min, ev_plus of the authors' table)
table = [
    ("GC-39",  1818,  7,  7, 7.55),
    ("GC-59",  1799, 15, 16, 4.91),
    ("GC-71",  1805,  8,  6, 2.57),
    ("GC-73",  1814,  3,  3, 6.77),
    ("GC-77",  1804,  6,  6, 0.40),
    ("GC-85",  1801,  6,  5, 2.26),
    ("GC-91",  1802, 10, 10, 1.55),
    ("GC-92",  1789,  7,  6, 1.94),
    ("GC-98",  1764, 14, 11, 3.59),
    ("GC-101", 1800, 14, 13, 4.77),
]
V_MEAN = 1803                                        # km/s, mean velocity of the galaxy

ids = [row[0] for row in table]
dv = np.array([row[1] - V_MEAN for row in table], float)            # velocity offsets
err_minus = np.array([row[2] for row in table], float)
err_plus = np.array([row[3] for row in table], float)
err = (err_minus + err_plus) / 2                     # one error per cluster (average of both sides)
R = np.array([row[4] for row in table], float)       # projected radii in kpc

print("Median radius:", np.median(R), "kpc   Largest radius:", R.max(), "kpc")


# %% 2. Three ways to measure the spread of the velocities ------------------------------
# Each function accepts one list of velocities, or a 2-D array with one list per row
# (this lets the Monte Carlo below do thousands of samples in a single call).

def biweight(x, c=9.0):
    """Biweight scale (Beers, Flynn & Gebhardt 1990): a spread that ignores outliers."""
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[-1]
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True)
    u = (x - med) / (c * mad)
    inside = np.abs(u) < 1                           # only points close to the median count
    top = np.sqrt(n * np.sum(np.where(inside, (x - med) ** 2 * (1 - u ** 2) ** 4, 0.0), axis=-1))
    bottom = np.abs(np.sum(np.where(inside, (1 - u ** 2) * (1 - 5 * u ** 2), 0.0), axis=-1))
    out = top / bottom
    return float(out[0]) if out.size == 1 else out


def nmad(x):
    """Normalized median absolute deviation: another outlier-resistant spread."""
    x = np.atleast_2d(np.asarray(x, float))
    med = np.median(x, axis=-1, keepdims=True)
    out = 1.4826 * np.median(np.abs(x - med), axis=-1)
    return float(out[0]) if out.size == 1 else out


def rms(x):
    """Ordinary root-mean-square spread about the mean: every point counts equally."""
    x = np.atleast_2d(np.asarray(x, float))
    out = np.sqrt(np.mean((x - x.mean(axis=-1, keepdims=True)) ** 2, axis=-1))
    return float(out[0]) if out.size == 1 else out


print("Observed spread (km/s):  biweight %.2f   nmad %.2f   rms %.2f" % (biweight(dv), nmad(dv), rms(dv)))


# %% 3. The mass estimator --------------------------------------------------------------
def tme_constant(alpha=0.0, gamma=0.9, beta=0.0):
    """Constant C in the TME (Eq. 4 of van Dokkum et al. 2018).
    alpha: slope of the potential, beta: orbit anisotropy, gamma: cluster density slope."""
    a = 4 * gamma_fn(alpha / 2 + 2.5) / (sqrt(pi) * gamma_fn(alpha / 2 + 1))
    b = (alpha + gamma + 1 - 2 * beta) / (alpha + 3 - beta * (alpha + 2))
    return a * b


def tme_mass(dv, r, r_out, sigma_intr, alpha=0.0, beta=0.0, gamma=0.9):
    """Mass inside r_out in Msun (Eq. 3 of van Dokkum et al. 2018).
    The velocities are rescaled so that their biweight spread equals sigma_intr
    (that is the correction for measurement errors)."""
    if sigma_intr <= 0:
        return 0.0                                   # no measurable dispersion -> no mass
    scale = sigma_intr / biweight(dv)
    spread = biweight(scale * np.asarray(dv, float) * np.asarray(r, float) ** (alpha / 2))
    return tme_constant(alpha, gamma, beta) / G * spread ** 2 * r_out ** (1 - alpha)


def wolf_mass(sigma, r_half):
    """Mass inside the half-number radius (Wolf et al. 2010): M = 4 sigma^2 r / G."""
    return 4 * sigma ** 2 * r_half / G


M_base = tme_mass(dv, R, R_OUT, SIGMA_BASE)
print("C for the baseline: %.3f" % tme_constant())
print("Baseline mass: %.2e Msun   (published < 3.4e8)" % M_base)
print("Mass inside the half-number radius: %.2e Msun   (published < 3.2e8)" % wolf_mass(SIGMA_BASE, R_HALF))


# %% 4. Intrinsic dispersion: what spread is left after removing the measurement errors? -
def sigma_quantile(dv, err, p, stat=biweight, nsim=2000, seed=42):
    """Find the true dispersion sigma for which the observed spread is as small as ours
    with probability p. We simulate nsim galaxies with true dispersion sigma, add
    the measurement errors, and measure the spread with `stat`.
    p = 0.5 -> best fit, p = 0.10 -> 90% upper limit, p = 0.84 / 0.16 -> 1-sigma range."""
    rng = np.random.default_rng(seed)
    n = len(dv)
    z1 = rng.standard_normal((nsim, n))              # random numbers, reused for every sigma
    z2 = rng.standard_normal((nsim, n))
    observed = stat(dv)

    def prob(sigma):
        fake = sigma * z1 + z2 * err                 # true motion + measurement error
        return np.mean(stat(fake) <= observed)

    if prob(0.0) < p:
        return 0.0                                   # errors alone already explain the spread
    return brentq(lambda s: prob(s) - p, 0.0, 80.0)  # solve prob(sigma) = p


def cumulative_curve(dv, err, grid, stat=biweight, nsim=2000, seed=42):
    """Probability that the true dispersion is below each value in `grid` (for plotting)."""
    rng = np.random.default_rng(seed)
    n = len(dv)
    z1 = rng.standard_normal((nsim, n))
    z2 = rng.standard_normal((nsim, n))
    observed = stat(dv)
    return np.array([1 - np.mean(stat(s * z1 + z2 * err) <= observed) for s in grid])


def gaussian_posterior(dv, err, smax=80.0, ns=2000):
    """Alternative method: probability of each dispersion given ALL ten velocities, assuming
    Gaussian scatter plus the measurement errors, flat priors, and a free mean velocity."""
    mu = np.linspace(-40, 40, 801)[:, None]          # possible mean velocities
    sig = np.linspace(0.01, smax, ns)[None, :]       # possible dispersions
    logL = np.zeros((mu.shape[0], sig.shape[1]))
    for v, e in zip(dv, err):
        var = sig ** 2 + e ** 2
        logL += -0.5 * np.log(2 * np.pi * var) - 0.5 * (v - mu) ** 2 / var
    post = np.exp(logL - logL.max())
    ps = post.sum(axis=0)                            # add up over the mean velocity
    ps /= ps.sum()
    s = sig[0]
    cdf = np.cumsum(ps)
    q = lambda p: float(s[np.searchsorted(cdf, p)])
    return {"grid": s, "cdf": cdf, "mode": float(s[np.argmax(ps)]), "median": q(0.5),
            "lo1": q(0.16), "hi1": q(0.84), "ul90": q(0.90)}


# %% 5. Tests of the sample: leave-one-out, bootstrap, and the joint Monte Carlo -----------
def leave_one_out(dv, err, R, ids):
    """Drop each cluster in turn and redo the 90% limit and the mass."""
    rows = []
    for k, name in enumerate(ids):
        keep = np.arange(len(dv)) != k
        s = sigma_quantile(dv[keep], err[keep], 0.10)
        M = tme_mass(dv[keep], R[keep], R[keep].max(), s)
        rows.append((name, s, R[keep].max(), M))
    return rows


def bootstrap_ul90(dv, err, R, nboot=500, nsim=1000, seed=1):
    """Resample the ten clusters with replacement and redo the 90% limit and the mass."""
    rng = np.random.default_rng(seed)
    n = len(dv)
    sig, mass = [], []
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        d, e, r = dv[idx], err[idx], R[idx]
        if np.median(np.abs(d - np.median(d))) == 0:
            continue                                  # skip a degenerate resample
        s = sigma_quantile(d, e, 0.10, nsim=nsim, seed=b)
        sig.append(s)
        mass.append(tme_mass(d, r, r.max(), s) if s > 0 else 0.0)
    return np.array(sig), np.array(mass)


def joint_mc(dv, R, sigma=SIGMA_BASE, n=20000, seed=3):
    """Pick alpha, beta and gamma at random from their allowed ranges, and return the mass
    divided by the baseline mass each time."""
    rng = np.random.default_rng(seed)
    M0 = tme_mass(dv, R, R_OUT, sigma)
    ratios, tried = [], 0
    while len(ratios) < n:
        a = rng.uniform(-1, 1)
        b = rng.uniform(-1, 0.75)
        g = rng.uniform(0.5, 1.5)
        tried += 1
        if (a + g + 1 - 2 * b) <= 0 or (a + 3 - b * (a + 2)) <= 0:
            continue                                  # these values give a non-positive constant
        ratios.append(tme_mass(dv, R, R_OUT, sigma, alpha=a, beta=b, gamma=g) / M0)
    return np.array(ratios), tried


# %% 6. Run everything and print the numbers used in the Results section -------------------
fmt = lambda x: "%.2e" % x

print("\n=== Intrinsic dispersion for each estimator (km/s) ===")
estimators = {"biweight": biweight, "nmad": nmad, "rms": rms}
est = {}
for name, stat in estimators.items():
    est[name] = {"best": sigma_quantile(dv, err, 0.5, stat),
                 "lo1": sigma_quantile(dv, err, 0.84, stat),
                 "hi1": sigma_quantile(dv, err, 0.16, stat),
                 "ul90": sigma_quantile(dv, err, 0.10, stat)}
    e = est[name]
    print("%-9s best %.1f   1-sigma (%.1f, %.1f)   90%% limit %.1f" %
          (name, e["best"], e["lo1"], e["hi1"], e["ul90"]))

gp = gaussian_posterior(dv, err)
print("likelihood: mode %.1f  median %.1f  16-84%% (%.1f, %.1f)  90%% bound %.1f" %
      (gp["mode"], gp["median"], gp["lo1"], gp["hi1"], gp["ul90"]))


def mass(sigma, **kw):
    """Shortcut: mass inside 7.6 kpc for a given dispersion."""
    return tme_mass(dv, R, R_OUT, sigma, **kw)


print("\n=== Mass for each estimator (M and M/Mstar) ===")
for name in ("biweight", "nmad", "rms"):
    Mb, Mu = mass(est[name]["best"]), mass(est[name]["ul90"])
    print("%-9s best fit: %s (%.2f)   90%% limit: %s (%.2f)" % (name, fmt(Mb), Mb / MSTAR, fmt(Mu), Mu / MSTAR))
M_mode, M_med, M_gp90 = mass(gp["mode"]), mass(gp["median"]), mass(gp["ul90"])
print("likelihood mode %s (%.2f)   median %s (%.2f)   90%% bound %s (%.2f)" %
      (fmt(M_mode), M_mode / MSTAR, fmt(M_med), M_med / MSTAR, fmt(M_gp90), M_gp90 / MSTAR))

print("\n=== Dynamical assumptions at sigma = 10.5 ===")
cases = {"alpha=-1": dict(alpha=-1), "alpha=+1": dict(alpha=1), "beta=-1": dict(beta=-1),
         "beta=0.5": dict(beta=0.5), "beta=0.75": dict(beta=0.75),
         "gamma=0.5": dict(gamma=0.5), "gamma=1.5": dict(gamma=1.5)}
dyn = {}
for label, kw in cases.items():
    dyn[label] = mass(SIGMA_BASE, **kw)
    print("%-10s mass %s   ratio to baseline %.2f" % (label, fmt(dyn[label]), dyn[label] / M_base))
ratios, tried = joint_mc(dv, R)
q16, q50, q84 = np.percentile(ratios, [16, 50, 84])
print("joint Monte Carlo: median %.2f, 16-84%% (%.2f, %.2f)" % (q50, q16, q84))

print("\n=== Leave-one-out ===")
loo = leave_one_out(dv, err, R, ids)
for name, s, r_out, M in loo:
    print("without %-7s sigma_90 %.1f  r_out %.2f  mass %s  M/Mstar %.2f" % (name, s, r_out, fmt(M), M / MSTAR))

print("\n=== Bootstrap ===")
sig_b, mass_b = bootstrap_ul90(dv, err, R)
print("usable resamples: %d, fraction with sigma_90 = 0: %.2f" % (len(sig_b), np.mean(sig_b == 0)))
print("mass percentiles 50 / 84 / 90: %s / %s / %s" %
      (fmt(np.percentile(mass_b, 50)), fmt(np.percentile(mass_b, 84)), fmt(np.percentile(mass_b, 90))))
print("fraction of resamples with mass above the stellar mass: %.2f" % np.mean(mass_b > MSTAR))

print("\n=== Comparison with the stellar mass (baseline) ===")
print("M / Mstar = %.2f   M - Mstar = %s Msun" % (M_base / MSTAR, fmt(M_base - MSTAR)))


# %% 7. Figure 1: the data and the dispersion limits --------------------------------------
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

left = ax[0]
left.errorbar(R, dv, yerr=[err_minus, err_plus], fmt="o", color="k", ms=4, capsize=2, lw=1)
for r_, v_, name in zip(R, dv, ids):
    if name in ("GC-98", "GC-39", "GC-73", "GC-91"):
        left.annotate(name.replace("GC-", ""), (r_, v_), textcoords="offset points", xytext=(5, 4), fontsize=7)
left.axhline(0, color="0.6", lw=0.8, ls=":")
left.axhline(SIGMA_BASE, color="C3", ls="--", lw=1, label=r"$\pm10.5$ km s$^{-1}$ (published 90% limit)")
left.axhline(-SIGMA_BASE, color="C3", ls="--", lw=1)
left.axhline(gp["ul90"], color="C0", ls="-.", lw=1, label=r"$\pm%.1f$ km s$^{-1}$ (likelihood 90%% bound)" % gp["ul90"])
left.axhline(-gp["ul90"], color="C0", ls="-.", lw=1)
left.set_xlabel("Projected radius $R$ (kpc)")
left.set_ylabel(r"$v - \langle v\rangle$ (km s$^{-1}$)")
left.set_ylim(-82, 45)
left.legend(loc="lower right", frameon=False, fontsize=8)
left.text(0.02, 0.96, "(a)", transform=left.transAxes, va="top", fontweight="bold")

right = ax[1]
grid = np.linspace(0, 40, 161)
right.plot(grid, cumulative_curve(dv, err, grid), color="C3", label="Biweight forward model")
keep = gp["grid"] <= 40
right.plot(gp["grid"][keep], gp["cdf"][keep], color="C0", label="Gaussian likelihood")
right.axhline(0.9, color="0.4", ls=":", lw=0.9)
right.axvline(est["biweight"]["ul90"], color="C3", ls="--", lw=0.9)
right.axvline(gp["ul90"], color="C0", ls="-.", lw=0.9)
right.set_xlabel(r"Intrinsic dispersion $\sigma$ (km s$^{-1}$)")
right.set_ylabel(r"Cumulative probability $P(\sigma_{\rm intr}<\sigma)$")
right.set_xlim(0, 40)
right.set_ylim(0, 1.02)
right.legend(loc="center right", frameon=False, fontsize=8)
right.text(0.02, 0.96, "(b)", transform=right.transAxes, va="top", fontweight="bold")
fig.tight_layout()
fig.savefig("figs/fig_dispersion.pdf")
plt.show()


# %% 8. Figure 2: how the mass changes with each assumption -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})

a = ax[0]
rows = [   # label, mass at the low end, mass at the high end, colour group
    ("Biweight (best fit, 90% limit)", mass(est["biweight"]["best"]), mass(est["biweight"]["ul90"]), "C0"),
    ("NMAD", mass(est["nmad"]["best"]), mass(est["nmad"]["ul90"]), "C0"),
    ("rms", mass(est["rms"]["best"]), mass(est["rms"]["ul90"]), "C0"),
    ("Gaussian likelihood (mode, 90%)", mass(gp["mode"]), mass(gp["ul90"]), "C0"),
    (r"$\alpha=-1$", dyn["alpha=-1"], dyn["alpha=-1"], "C2"),
    (r"$\alpha=+1$", dyn["alpha=+1"], dyn["alpha=+1"], "C2"),
    (r"$\beta=-1$ (tangential)", dyn["beta=-1"], dyn["beta=-1"], "C2"),
    (r"$\beta=0.5$ (radial)", dyn["beta=0.5"], dyn["beta=0.5"], "C2"),
    (r"$\gamma=0.5$", dyn["gamma=0.5"], dyn["gamma=0.5"], "C2"),
    (r"$\gamma=1.5$", dyn["gamma=1.5"], dyn["gamma=1.5"], "C2"),
    (r"Joint MC ($\alpha,\beta,\gamma$; 16-84%)", q16 * M_base, q84 * M_base, "C2"),
]
ypos = np.arange(len(rows))[::-1]
for y, (label, low, high, colour) in zip(ypos, rows):
    if low > 0 and abs(high - low) / high > 1e-6:
        a.plot([low / MSTAR, high / MSTAR], [y, y], color=colour, lw=2)
        a.plot(low / MSTAR, y, "o", mfc="white", mec=colour, ms=5)
    a.plot(high / MSTAR, y, "o", color=colour, ms=5)
a.axvline(1, color="k", ls=":", lw=1, label=r"$M_{\rm TME}=M_\star$ (stars only)")
a.axvline(3.4e8 / MSTAR, color="C3", ls="--", lw=1, label=r"Published limit ($3.4\times10^{8}\,M_\odot$)")
a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), frameon=False, fontsize=7)
a.set_yticks(ypos)
a.set_yticklabels([row[0] for row in rows])
a.set_xscale("log")
a.set_xlim(0.1, 12)
a.set_xticks([0.1, 0.3, 1, 3, 10])
a.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
a.set_xlabel(r"$M_{\rm TME}(<7.6\,{\rm kpc})\,/\,M_\star$")
a.text(0.02, 0.98, "(a)", transform=a.transAxes, va="top", fontweight="bold")

b = ax[1]
names = [row[0] for row in loo]
loo_mass = np.array([row[3] for row in loo])
yb = np.arange(len(names))[::-1]
b.plot(loo_mass / MSTAR, yb, "o", color="C1", ms=5)
b.axvline(M_base / MSTAR, color="k", ls="--", lw=1, label="All ten")
b.axvline(1, color="k", ls=":", lw=1, label="Stars only")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=7)
b.set_yticks(yb)
b.set_yticklabels(["without " + n for n in names])
b.set_xscale("log")
b.set_xlim(0.3, 12)
b.set_xticks([0.3, 1, 3, 10])
b.set_xticklabels(["0.3", "1", "3", "10"])
b.set_xlabel(r"$M_{\rm TME}$ (90% limit)$\,/\,M_\star$")
b.text(0.02, 0.98, "(b)", transform=b.transAxes, va="top", fontweight="bold")
fig.tight_layout()
fig.savefig("figs/fig_mass_summary.pdf", bbox_inches="tight")
plt.show()


# In[ ]:




