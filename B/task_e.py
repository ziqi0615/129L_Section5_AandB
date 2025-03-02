#!/usr/bin/env python

from numpy import linspace, exp
from matplotlib import pyplot as plt

particle_count = 100
beta_energy = linspace(0, 1, 100)
excited_state = 1 / (exp(beta_energy) - 1) - (1 + particle_count) / (exp(beta_energy * (1 + particle_count)) - 1)
excited_state[0] = particle_count / 2
ground_state = particle_count - excited_state

plt.plot(beta_energy, ground_state, label=r'$\left<n_0\right>$')
plt.plot(beta_energy, excited_state, label=r'$\left<n_\epsilon\right>$')
plt.xlabel(r'$\beta\epsilon$')
plt.title(f'$N={particle_count}$')
plt.legend()
plt.show()
