# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:29:20 2026

@author: leoho
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. NVIDIA Data Center Revenue
# -----------------------------
years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
revenue = np.array([1.93, 2.93, 2.98, 6.7, 10.61, 15.0, 36.2, 115.2, 193.74])  # billions USD

# -----------------------------
# 2. Logistic function
# -----------------------------
def logistic(t, r, t0, K):
    return K / (1 + np.exp(-r * (t - t0)))

# -----------------------------
# 3. Fit logistic curves for fixed K values
# -----------------------------
candidate_K = [300, 500, 700]  # billions
fits = {}

for K in candidate_K:
    # Fit r and t0 while keeping K fixed
    def model(t, r, t0):
        return logistic(t, r, t0, K)

    popt, pcov = curve_fit(model, years, revenue, p0=[0.5, 2024])
    r, t0 = popt
    fits[K] = {"r": r, "t0": t0}

    print(f"K = {K}B → r = {r:.4f}, t0 = {t0:.2f}")

# -----------------------------
# 4. Compute saturation year (90% of K)
# -----------------------------
def saturation_year(K, r, t0):
    # Solve logistic(t) = 0.9K
    # 0.9K = K / (1 + exp(-r(t - t0)))
    # Solve for t
    return t0 + (np.log(9) / r)

for K in candidate_K:
    r = fits[K]["r"]
    t0 = fits[K]["t0"]
    sat = saturation_year(K, r, t0)
    print(f"K = {K}B → Inflection year t0 ≈ {t0:.1f}, 90% saturation ≈ {sat:.1f}")

# -----------------------------
# 5. Generate curves for plotting
# -----------------------------
future_years = np.linspace(2018, 2035, 300)
predictions = {}

for K in candidate_K:
    r = fits[K]["r"]
    t0 = fits[K]["t0"]
    predictions[K] = logistic(future_years, r, t0, K)

# -----------------------------
# 6. Plot all three scenarios
# -----------------------------
plt.figure(figsize=(12, 7))
plt.scatter(years, revenue, color='black', label="Actual Revenue", s=60)

colors = {300: "blue", 500: "green", 700: "red"}

for K in candidate_K:
    plt.plot(future_years, predictions[K], 
             label=f"Logistic Fit (K={K}B)", 
             linewidth=2.2, color=colors[K])

plt.title("NVIDIA Data Center Revenue – Logistic Growth Scenarios")
plt.xlabel("Year")
plt.ylabel("Revenue (Billions USD)")
plt.legend()
plt.grid(True)
plt.show()
