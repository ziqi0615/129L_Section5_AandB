#!/usr/bin/env python

from numpy import linspace, exp
from matplotlib import pyplot as plt

beta_energy = linspace(0, 5, 100)
ground_state = 1 / (1 + exp(-beta_energy))
excited_state = 1 / (1 + exp(beta_energy))

plt.plot(beta_energy, ground_state, label=r'$\left<n_0\right>_{\mathrm{C}}/N$')
plt.plot(beta_energy, excited_state, label=r'$\left<n_\epsilon\right>_{\mathrm{C}}/N$')
plt.xlabel(r'$\beta\epsilon$')
plt.legend()
plt.show()
