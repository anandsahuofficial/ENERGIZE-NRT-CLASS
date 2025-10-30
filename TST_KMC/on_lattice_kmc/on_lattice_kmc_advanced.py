import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter


# Parameters
L = 200             # lattice size
N_steps = 5000      # total KMC steps
kT = 0.025         # thermal energy (eV) - increase temperature
steps_per_frame = 50
n_frames = N_steps // steps_per_frame

# Initialize lattice with 1% vacancies randomly
lattice = np.ones((L, L), dtype=int)
initial_vac_frac = 0.01
num_initial_vac = int(L*L*initial_vac_frac)
vac_indices = np.random.choice(L*L, size=num_initial_vac, replace=False)
for index in vac_indices:
    x, y = divmod(index, L)
    lattice[x, y] = 0

# Random vacancy formation energies (0.025 to 0.1 eV)
vac_formation_energies = np.random.uniform(0.025, 0.1, size=(L, L))

vacancy_counts = []
lattice_snapshots = []

def get_vacancy_energy(x, y):
    return vac_formation_energies[x, y]

def attempt_vacancy_change(x, y):
   '''

   TO-DO

   '''

   accept = False
   if lattice[x, y] == 1:  # If site is occupied
        delta_E = get_vacancy_energy(x, y)
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / kT):
            lattice[x, y] = 0  # Create vacancy
            accept = True 
        else:  # If site is vacant
            delta_E = -get_vacancy_energy(x, y)
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / kT):
                lattice[x, y] = 1  # Fill vacancy
                accept = True

   return accept

def perform_kmc_steps(n):
    for _ in range(n):
        '''
        
        TO-DO
        
        '''
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        attempt_vacancy_change(x, y)
# ==== KMC simulation and snapshot saving ====
        # if _ % steps_per_frame == 0:
        #     vac_count = np.sum(lattice == 0)
        #     vacancy_counts.append(vac_count)
        #     # Make a copy of lattice for animation snapshot
        #     lattice_snapshots.append(lattice.copy())


# ==== Perform full KMC simulation first and save snapshots ====

print("Running full KMC simulation...")
for frame in range(n_frames):
    perform_kmc_steps(steps_per_frame)
    vac_count = np.sum(lattice == 0)
    vacancy_counts.append(vac_count)
    # Make a copy of lattice for animation snapshot
    lattice_snapshots.append(lattice.copy())

print("KMC simulation done. Total frames:", len(lattice_snapshots))

# ==== Animate saved snapshots ====

fig, ax = plt.subplots(figsize=(6,6))
cmap = plt.colormaps.get_cmap('Greys_r')  # Vacancy=white, atom=black
im = ax.imshow(lattice_snapshots[0], cmap=cmap, vmin=0, vmax=1)
ax.axis('off')

def update(frame):
    im.set_data(lattice_snapshots[frame])
    ax.set_title(f'KMC Step {(frame+1)*steps_per_frame}\nVacancies: {vacancy_counts[frame]}')
    return [im]

anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True, interval=50)

# anim.save('vacancy_kmc_separate.mp4', writer='ffmpeg', fps=20)
writer = PillowWriter(fps=20)
anim.save('vacancy_kmc_separate.gif', writer=writer)
plt.close()

# ==== Plot vacancy counts ====

plt.figure(figsize=(8,5))
steps_array = np.arange(len(vacancy_counts)) * steps_per_frame
plt.plot(steps_array, vacancy_counts, marker='o')
plt.xlabel('KMC Steps')
plt.ylabel('Number of Vacancies')
plt.title('Vacancy concentration vs. KMC Step')
plt.grid(True)
plt.tight_layout()
plt.savefig('vacancy_count_separate.png')
plt.show()