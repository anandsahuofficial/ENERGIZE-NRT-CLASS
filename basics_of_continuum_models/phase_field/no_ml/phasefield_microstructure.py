#!/usr/bin/env python3
"""
Integrated script for 2D Allen–Cahn phase-field simulation with optional CNN training and ML acceleration.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow.keras import models, layers

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

# ----------------- CNN Training -----------------

def generate_training_data(N, Q, steps, dt, Lmob, W, lam, seed):
    rng = np.random.default_rng(seed)
    eps = max(1.5 / N, 0.01)
    eps2 = eps * eps
    k2 = make_k2(N, 1.0)
    phi = initialize_grains(N, Q, rng, 1.0)

    X = []
    Y = []
    for _ in range(steps):
        phi_next = ac_grain_step(phi, dt, Lmob, W, eps2, lam, k2)
        X.append(np.transpose(phi, (1,2,0)))
        Y.append(np.transpose(phi_next, (1,2,0)))
        phi = phi_next
    X = np.array(X); Y = np.array(Y)
    return X, Y

def train_cnn(X, Y, epochs=5, batch_size=16):
    N, _, Q = X.shape[1], X.shape[2], X.shape[3]
    model = models.Sequential([
        layers.Input(shape=(N,N,Q)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(Q, (3,3), activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    return model

# ----------------- Simulation with visualization -----------------

def simulate_with_ml(N, L, dt, steps, plot_interval, Q, Lmob, W, lam, seed, ml_model=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

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
            if ml_model is not None:
                phi_input = phi.transpose(1,2,0)[np.newaxis,...]
                phi_pred = ml_model.predict(phi_input, verbose=0)
                phi = phi_pred[0].transpose(2,0,1)
            else:
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
    parser = argparse.ArgumentParser(description='Integrated Allen–Cahn simulation with optional CNN training.')
    parser.add_argument('--mlsave', type=str, default='cnn_model.h5', help='Path to save trained CNN')
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--Q', type=int, default=8)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--Lmob', type=float, default=1.0)
    parser.add_argument('--W', type=float, default=5.0)
    parser.add_argument('--lam', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ml', type=str, default=None, help='Path to load CNN model (.h5)')
    args = parser.parse_args()

    ml_model = None
    train = True
    if train:
        print('Generating training data...')
        X, Y = generate_training_data(args.N, args.Q, args.steps, args.dt, args.Lmob, args.W, args.lam, args.seed)
        print('Training CNN...')
        ml_model = train_cnn(X, Y, epochs=50)
        ml_model.save(args.mlsave)
        print(f'Trained CNN saved to {args.mlsave}')
    elif args.ml is not None:
        ml_model = models.load_model(args.ml)
        print(f'Loaded CNN model from {args.ml}')

    simulate_with_ml(args.N, 1.0, args.dt, args.steps, plot_interval=1, Q=args.Q,
                     Lmob=args.Lmob, W=args.W, lam=args.lam, seed=args.seed, ml_model=ml_model)

if __name__ == '__main__':
    main()
