#!/usr/bin/env python

from numpy import linspace, array, exp, sum, geomspace
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def compute_quantities(energy_levels, beta_values, particle_count):
    chemical_potential_list = []
    ground_state_list = []
    d_ground_state_dT_list = []
    heat_capacity_list = []
    
    for beta in beta_values:
        chemical_potential = fsolve(lambda mu: sum(1 / (exp(beta * (energy_levels - mu)) - 1)) - particle_count, 
                                    energy_levels[0] - 1 / beta / particle_count)[0]
        energy_diff = energy_levels - chemical_potential
        exp_beta_energy = exp(beta * energy_diff)
        occupation_numbers = 1 / (exp_beta_energy - 1)
        derivative = exp_beta_energy * occupation_numbers**2
        d_occupation_dT = derivative * beta**2 * (energy_diff - sum(derivative * energy_diff) / sum(derivative))
        
        chemical_potential_list.append(chemical_potential)
        ground_state_list.append(occupation_numbers[0])
        d_ground_state_dT_list.append(d_occupation_dT[0])
        heat_capacity_list.append(sum(d_occupation_dT * energy_levels))
    
    return array(chemical_potential_list), array(ground_state_list), array(d_ground_state_dT_list), array(heat_capacity_list)

def plot_quantities(energy_levels, temperature_values, particle_count=1e5):
    beta_values = 1 / temperature_values
    chemical_potential, ground_state, d_ground_state_dT, heat_capacity = compute_quantities(energy_levels, beta_values, particle_count)
    
    fig, (ax_mu, ax_n0, ax_n0_log, ax_dn0_dT, ax_cv) = plt.subplots(5, sharex=True)
    
    ax_mu.plot(temperature_values, -chemical_potential)
    ax_mu.set_ylabel(r'$-\mu$')
    ax_mu.set_yscale('log')
    
    ax_n0.plot(temperature_values, ground_state)
    ax_n0.set_ylabel(r'$\left<n_0\right>$')
    
    ax_n0_log.plot(temperature_values, ground_state)
    ax_n0_log.set_ylabel(r'$\left<n_0\right>$')
    ax_n0_log.set_yscale('log')
    
    ax_dn0_dT.plot(temperature_values, -d_ground_state_dT)
    ax_dn0_dT.set_ylabel(r'$-\frac{\mathrm{d}\left<n_0\right>}{\mathrm{d}T}$')
    
    ax_cv.plot(temperature_values, heat_capacity)
    ax_cv.set_ylabel(r'$C_V$')
    
    ax_n0.set_xscale('log')
    plt.xlabel(r'T')
    plt.show()

plot_quantities(linspace(0, 1, 2), geomspace(1e1, 1e8, 100))
plot_quantities(linspace(0, 2000, 200)**(1/20), geomspace(1e-1, 1e6, 100))
