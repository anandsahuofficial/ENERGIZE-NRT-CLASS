import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def kinetic_operator_2d(Nx, Ny, dx, dy):
    N = Nx * Ny
    diag = np.ones(N) * 2 * (1 / dx**2 + 1 / dy**2)
    off_x = -1 / dx**2
    off_y = -1 / dy**2

    H = np.zeros((N, N))

    def idx(ix, iy):
        return iy * Nx + ix

    for iy in range(Ny):
        for ix in range(Nx):
            i = idx(ix, iy)
            H[i, i] = diag[i]

            if ix > 0:
                H[i, idx(ix - 1, iy)] = off_x
            if ix < Nx - 1:
                H[i, idx(ix + 1, iy)] = off_x
            if iy > 0:
                H[i, idx(ix, iy - 1)] = off_y
            if iy < Ny - 1:
                H[i, idx(ix, iy + 1)] = off_y

    return -0.5 * H

def potential_from_particles(x_grid, y_grid, particles, epsilon=1.0, sigma=0.2):
    V = np.zeros_like(x_grid)
    for (px, py) in particles:
        dx = x_grid - px
        dy = y_grid - py
        r = np.sqrt(dx**2 + dy**2) + 1e-8
        lj = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        V += lj - np.min(lj)
    return V

def lj_pairwise_forces(particles, epsilon=1.0, sigma=0.2):
    N = len(particles)
    forces = np.zeros_like(particles)

    for i in range(N):
        for j in range(i + 1, N):
            rij = particles[i] - particles[j]
            r = np.linalg.norm(rij) + 1e-8
            f_mag = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
            f_vec = f_mag * (rij / r)
            forces[i] += f_vec
            forces[j] -= f_vec
    return forces

def confining_force(particles, Lx, Ly, k_wall=50.0, margin=0.3):
    forces = np.zeros_like(particles)
    for i, (x, y) in enumerate(particles):
        if x < margin:
            forces[i, 0] += k_wall * (margin - x)
        if x > Lx - margin:
            forces[i, 0] -= k_wall * (x - (Lx - margin))
        if y < margin:
            forces[i, 1] += k_wall * (margin - y)
        if y > Ly - margin:
            forces[i, 1] -= k_wall * (y - (Ly - margin))
    return forces

def scf_step_2d(H):
    eigvals, eigvecs = eigh(H)
    psi = eigvecs[:, 0]
    psi /= np.linalg.norm(psi)
    energy = eigvals[0]
    return energy, psi

def energy_and_forces(particles, x_grid, y_grid, kinetic_op, epsilon, sigma, dx, dy):
    V_2d = potential_from_particles(x_grid, y_grid, particles, epsilon, sigma)
    Nx, Ny = x_grid.shape
    V_diag = V_2d.flatten()
    H = kinetic_op + np.diag(V_diag)

    energy, psi = scf_step_2d(H)

    forces_scf = np.zeros_like(particles)
    h = 0.1

    for i, (x0, y0) in enumerate(particles):
        for dim in [0, 1]:
            shift = np.zeros_like(particles)
            shift[i, dim] = h

            V_plus = potential_from_particles(x_grid, y_grid, particles + shift, epsilon, sigma).flatten()
            H_plus = kinetic_op + np.diag(V_plus)
            e_plus, _ = scf_step_2d(H_plus)

            V_minus = potential_from_particles(x_grid, y_grid, particles - shift, epsilon, sigma).flatten()
            H_minus = kinetic_op + np.diag(V_minus)
            e_minus, _ = scf_step_2d(H_minus)

            forces_scf[i, dim] = -(e_plus - e_minus) / (2 * h)

    forces_classical = lj_pairwise_forces(particles, epsilon, sigma)
    forces_confine = confining_force(particles, x_grid.max(), y_grid.max())

    forces = forces_scf + forces_classical + forces_confine

    return energy, forces, psi

def geometry_optimization(particles_init, x_grid, y_grid, kinetic_op,
                          epsilon, sigma, dx, dy,
                          max_steps=50, tol=1e-4, step_size=0.005):
    particles = particles_init.copy()
    energies = []
    forces_norm = []
    positions = [particles.copy()]

    for step in range(max_steps):
        # get energies, forces, psi
        '''
        STUDENT PORTION
        '''



        # Clip forces to max norm 1.0
        max_force_norm = 1.0
        for i in range(len(forces)):
            norm = np.linalg.norm(forces[i])
            if norm > max_force_norm:
                forces[i] = (forces[i] / norm) * max_force_norm


        # determine if force tolerance is satisfied
        '''
        STUDENT PORTION
        '''




        # update positions
        '''
        STUDENT PORTION
        '''



        # bound particles inside simulation box
        particles[:, 0] = np.clip(particles[:, 0], 0, x_grid.max())
        particles[:, 1] = np.clip(particles[:, 1], 0, y_grid.max())
        positions.append(particles.copy())

    return particles, energies, forces_norm, positions, psi

def main():
     # Setup grid, kinetic operator, etc.
    Nx, Ny = 50, 50
    Lx, Ly = 5.0, 5.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    kinetic_op = kinetic_operator_2d(Nx, Ny, dx, dy)

    # Initial dimer positions
    initial_bond_length = 3.0
    center_y = Ly / 2
    particle1 = np.array([Lx / 2 - initial_bond_length / 2, center_y])
    particle2 = np.array([Lx / 2 + initial_bond_length / 2, center_y])
    particles_init = np.array([particle1, particle2])

    # Run geometry optimization
    epsilon = 1.0
    sigma = 1.0
    particles_opt, energies, forces_norm, positions, psi = geometry_optimization(
        particles_init, x_grid, y_grid, kinetic_op,
        epsilon, sigma, dx, dy,
        max_steps=200, tol=1e-4, step_size=0.01)

    print("Optimized dimer positions:\n", particles_opt)
    print(f"Optimized bond length: {np.linalg.norm(particles_opt[1] - particles_opt[0]):.4f}")

    # Compute bond distances and LJ potential energies during GO
    bond_distances = []
    lj_potential_energies = []
    for pos in positions:
        d = np.linalg.norm(pos[1] - pos[0])
        bond_distances.append(d)
        # Lennard-Jones potential energy only for this bond
        r = d + 1e-8
        lj_e = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        lj_potential_energies.append(lj_e)

    # Plot energy, sum of forces, and potential energy vs bond distance in 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(energies, '-o')
    axs[0].set_ylabel("Total Energy")
    axs[0].set_title("Energy vs GO iteration")
    axs[0].grid(True)

    axs[1].plot(forces_norm, '-o', color='r')
    axs[1].set_ylabel("Sum of forces")
    axs[1].set_title("Sum of forces vs GO iteration")
    axs[1].grid(True)

    axs[2].plot(bond_distances, '-o', color='g')
    axs[2].set_xlabel("GO iteration")
    axs[2].set_ylabel("Bond distance")
    axs[2].set_title("Bond distance vs GO iteration during geometry optimization")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()