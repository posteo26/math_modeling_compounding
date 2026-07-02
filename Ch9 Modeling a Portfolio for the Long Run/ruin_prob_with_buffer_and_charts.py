# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ============================================================
# 1. Model Parameters
# ============================================================

mu     = 0.06      # main account expected return
sigma  = 0.15      # main account volatility
r      = 0.02      # buffer interest rate
c      = 0.04      # annual withdrawal (fraction of initial wealth)

W0     = 1.0       # initial main account (normalized)
C0     = 0.20      # initial buffer (20% of initial wealth)

T_years = 40
steps_per_year = 252
dt = 1.0 / steps_per_year
n_steps = int(T_years * steps_per_year)

n_paths = 50_000


# ============================================================
# 2. Single Path Simulation with Buffer
# ============================================================

def simulate_path(mu, sigma, r, c, W0, C0, dt, n_steps, rng):
    W = W0
    C = C0
    withdrawal = c * W0 * dt

    for _ in range(n_steps):

        # Ruin if both accounts empty
        if W <= 0 and C <= 0:
            return True

        # Withdraw from buffer first
        if C >= withdrawal:
            C -= withdrawal
        else:
            remaining = withdrawal - C
            C = 0
            W -= remaining

        # Grow buffer (safe)
        if C > 0:
            C += r * C * dt

        # Grow main account (risky)
        if W > 0:
            z = rng.normal()
            dW = mu * W * dt + sigma * W * np.sqrt(dt) * z
            W += dW

    return (W <= 0 and C <= 0)


# ============================================================
# 3. Estimate Ruin Probability
# ============================================================

def estimate_rp(mu, sigma, r, c, W0, C0, dt, n_steps, n_paths, rng):
    ruins = 0
    for _ in range(n_paths):
        if simulate_path(mu, sigma, r, c, W0, C0, dt, n_steps, rng):
            ruins += 1
    return ruins / n_paths

rp = estimate_rp(mu, sigma, r, c, W0, C0, dt, n_steps, n_paths, rng)
print(f"Estimated ruin probability with buffer: {rp:.3f}")


# ============================================================
# 4. Visualize Sample Paths (Main + Buffer)
# ============================================================

def simulate_paths(mu, sigma, r, c, W0, C0, dt, n_steps, n_paths_plot, rng):
    withdrawal = c * W0 * dt
    W_paths = np.zeros((n_paths_plot, n_steps + 1))
    C_paths = np.zeros((n_paths_plot, n_steps + 1))
    t_grid = np.linspace(0, n_steps * dt, n_steps + 1)

    for i in range(n_paths_plot):
        W = W0
        C = C0
        W_paths[i, 0] = W
        C_paths[i, 0] = C

        for k in range(1, n_steps + 1):

            # Withdraw
            if C >= withdrawal:
                C -= withdrawal
            else:
                remaining = withdrawal - C
                C = 0
                W -= remaining

            # Buffer growth
            if C > 0:
                C += r * C * dt

            # Main growth
            if W > 0:
                z = rng.normal()
                dW = mu * W * dt + sigma * W * np.sqrt(dt) * z
                W += dW

            W_paths[i, k] = max(W, 0)
            C_paths[i, k] = max(C, 0)

    return t_grid, W_paths, C_paths


# Plot sample paths
t, Wp, Cp = simulate_paths(mu, sigma, r, c, W0, C0, dt, n_steps, 50, rng)

plt.figure(figsize=(10, 6))
for i in range(Wp.shape[0]):
    plt.plot(t, Wp[i], color='steelblue', alpha=0.3)
plt.axhline(0, color='black', linewidth=1)
plt.title("Main Account Sample Paths")
plt.xlabel("Time (years)")
plt.ylabel("Main Account Wealth")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
for i in range(Cp.shape[0]):
    plt.plot(t, Cp[i], color='darkgreen', alpha=0.3)
plt.axhline(0, color='black', linewidth=1)
plt.title("Buffer Account Sample Paths")
plt.xlabel("Time (years)")
plt.ylabel("Buffer Wealth")
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 5. Sensitivity Analysis: Ruin Probability vs Withdrawal Rate
# ============================================================

withdrawal_grid = np.linspace(0.02, 0.08, 7)
ruin_probs = []

print("\nWithdrawal Rate Sensitivity:")
for c_test in withdrawal_grid:
    rp = estimate_rp(mu, sigma, r, c_test, W0, C0, dt, n_steps, n_paths, rng)
    ruin_probs.append(rp)
    print(f"c = {c_test:.3f}, ruin probability = {rp:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(withdrawal_grid, ruin_probs, marker='o')
plt.xlabel("Withdrawal rate (fraction of initial wealth per year)")
plt.ylabel("Ruin probability over 40 years")
plt.title("Ruin Probability vs Withdrawal Rate (with Buffer)")
plt.grid(True, alpha=0.3)
plt.show()

print("\nSimulation complete.")
