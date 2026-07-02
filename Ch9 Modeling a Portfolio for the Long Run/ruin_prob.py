# -*- coding: utf-8 -*-

# ============================================================
#  Ruin Probability Simulation for Wealth Diffusion with Withdrawals
#  Compatible with Spyder (no notebook cells required)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
rng = np.random.default_rng(42)

# ============================================================
# 1. Model Parameters
# ============================================================

mu     = 0.06      # expected annual return
sigma  = 0.15      # annual volatility
c      = 0.04      # annual withdrawal (fraction of initial wealth)
W0     = 1.0       # initial wealth (normalized)

T_years = 40
steps_per_year = 252
dt = 1.0 / steps_per_year
n_steps = int(T_years * steps_per_year)

n_paths = 50_000   # Monte Carlo paths

# ============================================================
# Parameter Validation (Warn if values look incorrect)
# ============================================================

def warn_if_invalid(name, value):
    """
    Warn if a parameter looks like it was entered incorrectly.
    Example: 5 instead of 0.05, 10 instead of 0.10, etc.
    """
    if value > 1.0:
        print(f"WARNING: {name} = {value} looks too large.")
        print("         Did you mean", value/100, "instead of", value, "?")
        print("         Rates must be decimals: 5% → 0.05, 10% → 0.10\n")

    if value < 0:
        print(f"WARNING: {name} = {value} is negative. Check your input.\n")

# Validate parameters
warn_if_invalid("mu (expected return)", mu)
warn_if_invalid("sigma (volatility)", sigma)
warn_if_invalid("c (withdrawal rate)", c)


# ============================================================
# 2. Single Path Simulation (Euler–Maruyama)
# ============================================================

def simulate_path(mu, sigma, c, W0, dt, n_steps, rng):
    """
    Simulate one wealth path with constant withdrawal c * W0 per year.
    Ruin occurs when W <= 0.
    """
    W = W0
    withdrawal = c * W0 * dt

    for _ in range(n_steps):
        if W <= 0.0:
            return True  # ruined

        z = rng.normal()
        dW = mu * W * dt + sigma * W * np.sqrt(dt) * z - withdrawal
        W += dW

    return W <= 0.0


# ============================================================
# 3. Estimate Ruin Probability
# ============================================================

def estimate_ruin_probability(mu, sigma, c, W0, dt, n_steps, n_paths, rng):
    ruins = 0
    for _ in range(n_paths):
        if simulate_path(mu, sigma, c, W0, dt, n_steps, rng):
            ruins += 1
    return ruins / n_paths

ruin_prob = estimate_ruin_probability(mu, sigma, c, W0, dt, n_steps, n_paths, rng)
print(f"Estimated ruin probability over {T_years} years: {ruin_prob:.3f}")


# ============================================================
# 4. Visualize Sample Paths
# ============================================================

def simulate_paths(mu, sigma, c, W0, dt, n_steps, n_paths_plot, rng):
    withdrawal = c * W0 * dt
    paths = np.zeros((n_paths_plot, n_steps + 1))
    t_grid = np.linspace(0, n_steps * dt, n_steps + 1)

    for i in range(n_paths_plot):
        W = W0
        paths[i, 0] = W
        for k in range(1, n_steps + 1):
            if W <= 0.0:
                W = 0.0
            else:
                z = rng.normal()
                dW = mu * W * dt + sigma * W * np.sqrt(dt) * z - withdrawal
                W += dW
            paths[i, k] = W

    return t_grid, paths

# Plot 50 sample paths
t_grid, paths = simulate_paths(mu, sigma, c, W0, dt, n_steps, 50, rng)

plt.figure(figsize=(10, 6))
for i in range(paths.shape[0]):
    plt.plot(t_grid, paths[i], color='steelblue', alpha=0.3)
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Time (years)")
plt.ylabel("Wealth")
plt.title("Sample Wealth Paths with Withdrawals")
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 5. Sensitivity Analysis: Ruin Probability vs Withdrawal Rate
# ============================================================

withdrawal_grid = np.linspace(0.02, 0.08, 7)  # 2% to 8%
ruin_probs = []

print("\nWithdrawal Rate Sensitivity:")
for c_test in withdrawal_grid:
    rp = estimate_ruin_probability(mu, sigma, c_test, W0, dt, n_steps, n_paths, rng)
    ruin_probs.append(rp)
    print(f"c = {c_test:.3f}, ruin probability = {rp:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(withdrawal_grid, ruin_probs, marker='o')
plt.xlabel("Withdrawal rate (fraction of initial wealth per year)")
plt.ylabel("Ruin probability over 40 years")
plt.title("Ruin Probability vs Withdrawal Rate")
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 6. Done
# ============================================================

print("\nSimulation complete.")
