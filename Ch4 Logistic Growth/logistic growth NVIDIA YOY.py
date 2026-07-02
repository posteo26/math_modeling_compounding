# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:35:07 2026

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
# 6. Compute YoY growth for each K
# -----------------------------
def yoy_growth(series, years):
    growth = []
    for i in range(1, len(series)):
        g = (series[i] / series[i-1]) - 1
        growth.append((years[i], g))
    return growth

print("\n--- YoY Growth by Scenario ---")
for K in candidate_K:
    curve = logistic(years, fits[K]["r"], fits[K]["t0"], K)
    growth = yoy_growth(curve, years)
    print(f"\nK = {K}B YoY Growth:")
    for yr, g in growth:
        print(f"  {yr}: {g*100:.1f}%")

# -----------------------------
# 7. Compute forward CAGR from each year to saturation
# -----------------------------
def forward_cagr(current_value, future_value, years_forward):
    return (future_value / current_value)**(1/years_forward) - 1

print("\n--- Forward CAGR to Saturation ---")
for K in candidate_K:
    r = fits[K]["r"]
    t0 = fits[K]["t0"]
    sat_year = saturation_year(K, r, t0)
    sat_value = 0.9 * K  # 90% saturation

    print(f"\nK = {K}B:")
    for i, yr in enumerate(years):
        current = revenue[i]
        years_forward = sat_year - yr
        if years_forward > 0:
            cagr = forward_cagr(current, sat_value, years_forward)
            print(f"  From {yr} to {sat_year:.1f}: CAGR ≈ {cagr*100:.1f}%")

# -----------------------------
# 8. Plot all three scenarios
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