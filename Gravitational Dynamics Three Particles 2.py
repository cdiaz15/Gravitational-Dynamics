import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

m1 = 200000000
m2 = 200000000
m3 = 200000000

x1 = 0
y1 = 0
x2 = 10
y2 = 0
x3 = 20
y3 = 20

velocity_x1 = 0
velocity_y1 = 0
velocity_x2 = 0
velocity_y2 = 0
velocity_x3 = 0
velocity_y3 = 0

particles_positions = np.full((1, 6), (x1, y1, x2, y2, x3, y3))
particles_velocities = np.full((1, 6), (velocity_x1, velocity_y1, velocity_x2, velocity_y2, velocity_x3, velocity_y3))

t = 0
n = 0

kinetic_energy_1_sum = 0
kinetic_energy_2_sum = 0
kinetic_energy_3_sum = 0
potential_energy_1_sum = 0
potential_energy_2_sum = 0
potential_energy_3_sum = 0

time_list = [t]

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def acceleration(m1, m2, m, x, soften):
    return ((6.674e-11 * m1 * m2) / (x ** 2 + soften)) / m


def kinetic_energy(m, v):
    return 0.5 * m * v * v


def potential_energy_f(m1, m2, x):
    return (- 6.674e-11 * m1 * m2) / x


def soften_f(x):
    return math.exp(- x / 5)


velocity_1 = math.sqrt((velocity_x1 ** 2) + (velocity_y1 ** 2))
velocity_2 = math.sqrt((velocity_x2 ** 2) + (velocity_y2 ** 2))
velocity_3 = math.sqrt((velocity_x3 ** 2) + (velocity_y3 ** 2))

kinetic_energy_initial_1 = kinetic_energy(m1, velocity_1)
kinetic_energy_initial_2 = kinetic_energy(m2, velocity_2)
kinetic_energy_initial_3 = kinetic_energy(m3, velocity_3)
kinetic_energies = np.full((1, 3), (kinetic_energy_initial_1, kinetic_energy_initial_2, kinetic_energy_initial_3))

distance_1_to_2 = distance(x1, y1, x2, y2)
distance_1_to_3 = distance(x1, y1, x3, y3)
distance_2_to_3 = distance(x2, y2, x3, y3)

potential_energy_initial_1 = potential_energy_f(m1, m2, distance_1_to_2) + potential_energy_f(m1, m3, distance_1_to_3)
potential_energy_initial_2 = potential_energy_f(m2, m1, distance_1_to_2) + potential_energy_f(m2, m3, distance_2_to_3)
potential_energy_initial_3 = potential_energy_f(m3, m1, distance_1_to_3) + potential_energy_f(m3, m2, distance_2_to_3)
potential_energies = np.full((1, 3), (potential_energy_initial_1, potential_energy_initial_2, potential_energy_initial_3)

Frame = pd.DataFrame(particles_positions)

for loop in range(2000):
    ts = 1
    distance_1_to_2 = distance(x1, y1, x2, y2)
    distance_1_to_3 = distance(x1, y1, x3, y3)
    distance_2_to_3 = distance(x2, y2, x3, y3)

    soften_1_to_2 = soften_f(distance_1_to_2)
    soften_1_to_3 = soften_f(distance_1_to_3)
    soften_2_to_3 = soften_f(distance_2_to_3)