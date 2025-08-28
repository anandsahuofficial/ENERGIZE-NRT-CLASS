#!/usr/bin/env python3
"""
2D Allen–Cahn phase-field simulation without any machine learning.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------- Utilities -----------------

def make_k2(N, L):
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    return np.add.outer(ky**2, kx**2)

# ----------------- Allen–Cahn -----------------

def initialize_grains(N, Q, rng, L):
    ys, xs = np.mgrid[0:N, 0:N]
    xs = xs * (L/N); ys = ys * (L/N)
    sx = rng.uniform(0, L, size=Q)
    sy = rng.uniform(0, L, size=Q)
    d2 = np.zeros((Q, N, N))
    for i in range(Q):
        dx = np.minimum(np.abs(xs - sx[i]), L - np.abs(xs - sx[i]))
        dy = np.minimum(np.abs(ys - sy[i]), L - np.abs(ys - sy[i]))
        d2[i] = dx*dx + dy*dy
    nearest = np.argmin(d2, axis=0)
    phi = np.zeros((Q, N, N), dtype=float)
    for i in range(Q): phi[i] = (nearest == i).astype(float)
    phi += 0.05 * rng.standard_normal((Q, N, N))
    phi = np.clip(phi, 0.0, 1.0)
    S = np.sum(phi, axis=0, keepdims=True) + 1e-12
    phi = phi / S
    return phi

def ac_grain_step(phi, dt, Lmob, W, eps2, lam, k2):
    Q, N, _ = phi.shape
    S = np.sum(phi, axis=0)
    dPsi = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
    penalty = 2.0 * lam * (S - 1.0)
    denom = 1.0 + dt * Lmob * eps2 * k2
    phi_new = np.empty_like(phi)
    for i in range(Q):
        rhs_real = W * dPsi[i] + penalty
        numer_hat = np.fft.fft2(phi[i]) - dt * Lmob * np.fft.fft2(rhs_real)
        phi_new[i] = np.real(np.fft.ifft2(numer_hat / denom))
    phi_new = np.clip(phi_new, 0.0, 1.0)
    S = np.sum(phi_new, axis=0, keepdims=True) + 1e-12
    phi_new = phi_new / S
    return phi_new

# ----------------- Simulation and visualization -----------------

def simulate_grains(N, L, dt, steps, plot_interval, Q, Lmob, W, lam, seed):
    rng = np.random.default_rng(seed)
    eps = max(1.5 / N, 0.01)
    eps2 = eps * eps
    dt = min(dt, 0.2 * eps2 / Lmob)
    k2 = make_k2(N, L)
    phi = initialize_grains(N, Q, rng, L)

    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
    labels = np.argmax(phi, axis=0)
    im = ax.imshow(labels, origin='lower', extent=[0, L, 0, L], interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))
    t = 0.0

    def update(_):
        nonlocal phi, t
        for _ in range(plot_interval):
            phi = ac_grain_step(phi, dt, Lmob, W, eps2, lam, k2)
            t += dt
        labels = np.argmax(phi, axis=0)
        im.set_data(labels)
        im.set_clim(0, Q-1)
        interfacial = float(np.sum(phi*(1-phi)))/(N*N)
        text.set_text(f't={t:.3f}  proxy γ={interfacial:.4f}')
        return im, text

    anim = FuncAnimation(fig, update, frames=steps//plot_interval, interval=1, blit=False, repeat=False)
    plt.show()

# ----------------- Main driver -----------------

def main():
    parser = argparse.ArgumentParser(description='2D Allen–Cahn phase-field simulation')
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--Q', type=int, default=8)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--Lmob', type=float, default=1.0)
    parser.add_argument('--W', type=float, default=5.0)
    parser.add_argument('--lam', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    simulate_grains(args.N, 1.0, args.dt, args.steps, plot_interval=1, Q=args.Q,
                    Lmob=args.Lmob, W=args.W, lam=args.lam, seed=args.seed)

if __name__ == '__main__':
    main()
