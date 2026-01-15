"""
ANJ-V001 (TIC 258775356) — TESS Time-Domain Analysis
Data: TESS SPOC PDCSAP light curves (Sectors 73, 74, 75, 78, 79, 81, 82, 83)
Outputs: periodogram.png, fold_2P.png, JD_lightcurve_mag.png
Results: P = 1.16385 d; Epoch (Min I, BJD_TDB) = 2460285.8135; Amp(TESS) ≈ 0.144 mag
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from lightkurve import search_lightcurve
tic = "258775356"
search = search_lightcurve(f"TIC {tic}", mission="TESS")
search_spoc = search[search.table["provenance_name"] == "SPOC"]
print("Total entries:", len(search))
print("SPOC entries :", len(search_spoc))
print(search_spoc)
lcs = search_spoc.download_all(download_dir="tess_lcs")
lc = lcs.stitch().remove_nans()
lc = lc[lc.quality == 0]
lc = lc.remove_outliers(sigma=5)
lc = lc.normalize()

t_btjd = lc.time.value
flux_norm = lc.flux.value

m = np.isfinite(t_btjd) & np.isfinite(flux_norm) & (flux_norm > 0)
t_btjd = t_btjd[m]
flux_norm = flux_norm[m]

mag_rel = -2.5 * np.log10(flux_norm)

mag_05 = np.percentile(mag_rel, 5)
mag_95 = np.percentile(mag_rel, 95)
amp_mag = mag_95 - mag_05

mag_min = np.percentile(mag_rel, 1)
mag_max = np.percentile(mag_rel, 99)
range_mag = mag_max - mag_min

print("\n--- Magnitude results (TESS relative) ---")
print("Amp (5-95%)  =", amp_mag, "mag")
print("Range (1-99%)=", range_mag, "mag")
print("Bright (min) =", mag_min, "mag")
print("Faint  (max) =", mag_max, "mag")

t_bjd = t_btjd + 2457000.0

plt.figure(figsize=(12,4))
plt.scatter(t_bjd, mag_rel, s=2)
plt.gca().invert_yaxis()
plt.xlabel("Time (BJD_TDB)")
plt.ylabel("Relative magnitude (TESS)")
plt.title("TESS SPOC stitched light curve (relative magnitudes)")
plt.savefig("JD_lightcurve_mag.png", dpi=220, bbox_inches="tight")
plt.close()

print("\n--- Epoch calculation block reached ---")
P = 1.16385
folded = lc.fold(P)
ph = folded.phase.value
fl = folded.flux.value
t = lc.time.value
imin = np.argmin(fl)
phase_min = ph[imin] % 1.0
epoch_btjd = t[0] + phase_min * P
epoch_bjd = epoch_btjd + 2457000.0
print("Phase(min) =", phase_min)
print("Epoch BTJD  =", epoch_btjd)
print("Epoch BJD   =", epoch_bjd)

print("Time span (days):", lc.time.max().value - lc.time.min().value)
t = lc.time.value
y = lc.flux.value
freq = np.linspace(0.05, 20.0, 100000)
ls = LombScargle(t, y)
power = ls.power(freq)
best_freq = freq[np.argmax(power)]
best_period = 1.0 / best_freq
print("\nBest period from LS =", best_period, "days")
plt.figure(figsize=(10,4))
plt.plot(1/freq, power)
plt.axvline(best_period, linestyle="--")
plt.xlim(0, 2)
plt.xlabel("Period (days)")
plt.ylabel("LS Power")
plt.title(f"TIC {tic} Periodogram (stitched sectors)")
plt.savefig("periodogram.png", dpi=200, bbox_inches="tight")
def plot_fold(P, fname):
    folded = lc.fold(P)
    plt.figure(figsize=(10,4))
    ph = folded.phase.value
    fl = folded.flux.value
    plt.scatter(ph, fl, s=4)
    plt.scatter(ph + 1, fl, s=4)
    plt.xlim(0,2)
    plt.xlabel("Phase (0–2)")
    plt.ylabel("Normalized flux")
    plt.title(f"Folded at P = {P:.6f} d")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
plot_fold(best_period, "fold_P.png")
plot_fold(2*best_period, "fold_2P.png")
print("\nSaved: periodogram.png, fold_P.png, fold_2P.png")

if __name__ == "__main__":
    main()
