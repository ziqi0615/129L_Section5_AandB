#!/usr/bin/env python

from numpy import cos, sin, sqrt, pi, dot, roll, linspace, hstack
from numpy.random import rand, seed
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from matplotlib import animation

seed(1108)

mass1, mass2, length1, length2, gravity = 0.5, 0.7, 0.3, 0.4, 9.8

def equations(t, state):
    angle1, angle2, momentum1, momentum2 = state
    denominator = mass2 * (mass1 + mass2 - mass2 * cos(angle1 - angle2)**2) * length1**2 * length2**2
    d_denominator = 2 * mass2**2 * length1**2 * length2**2 * cos(angle1 - angle2) * sin(angle1 - angle2)
    numerator = mass2 * length2**2 * momentum1**2 / 2 + (mass1 + mass2) * length1**2 * momentum2**2 / 2 - mass2 * length1 * length2 * momentum1 * momentum2 * cos(angle1 - angle2)
    d_numerator = mass2 * length1 * length2 * momentum1 * momentum2 * sin(angle1 - angle2)
    return [
        (mass2 * length2**2 * momentum1 - mass2 * length1 * length2 * cos(angle1 - angle2) * momentum2) / denominator,
        ((mass1 + mass2) * length1**2 * momentum2 - mass2 * length1 * length2 * cos(angle1 - angle2) * momentum1) / denominator,
        -(d_numerator * denominator - numerator * d_denominator) / denominator**2 - (mass1 + mass2) * gravity * length1 * sin(angle1),
        -(d_numerator * denominator - numerator * d_denominator) / denominator**2 - mass2 * gravity * length2 * sin(angle2),
    ]

# Solve equations
solution = solve_ivp(
    equations, (0, 5),
    y0=[1, 0.1, 0, 0],
    vectorized=True,
)

plt.plot(solution.y[1], solution.y[3])
plt.xlabel(r'$\theta_2$')
plt.ylabel(r'$p_2$')
plt.show()

plt.plot(solution.y[0], solution.y[1])
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.show()

# Animation function
def animate_motion(solution):
    pos_x1 = length1 * sin(solution.y[0])
    pos_y1 = -length1 * cos(solution.y[0])
    pos_x2 = pos_x1 + length2 * sin(solution.y[1])
    pos_y2 = pos_y1 - length2 * cos(solution.y[1])
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2, markersize=8)
    
    def update(frame):
        line.set_data([0, pos_x1[frame], pos_x2[frame]], [0, pos_y1[frame], pos_y2[frame]])
        return line,
    
    anim = animation.FuncAnimation(fig, update, frames=len(solution.t), interval=100, blit=True)
    plt.show()

animate_motion(solution)

# Phase space area calculation
def random_disk_sample():
    radius = sqrt(rand())
    angle = 2 * pi * rand()
    return radius * cos(angle), radius * sin(angle)

def compute_area(points):
    vertices = ConvexHull(points).points
    x_vals, y_vals = vertices[:, 0], vertices[:, 1]
    return abs(dot(x_vals, roll(y_vals, 1)) - dot(y_vals, roll(x_vals, 1))) / 2

num_samples = 20
initial_conditions = []
time_points = linspace(0, 5, 100)

for _ in range(num_samples):
    delta_angle2, delta_momentum2 = random_disk_sample()
    delta_angle2 *= 0.01
    delta_momentum2 *= 0.01
    initial_conditions.append([1, 0.1 + delta_angle2, 0, delta_momentum2])

solutions = [solve_ivp(equations, (0, 5), init_cond, vectorized=True, t_eval=time_points) for init_cond in initial_conditions]
computed_areas = [compute_area([[sol.y[1][i], sol.y[3][i]] for sol in solutions]) for i in range(len(time_points))]

plt.plot(time_points, computed_areas)
plt.xlabel(r'$t$')
plt.ylabel('Area')
plt.show()
