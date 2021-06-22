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
y2 = 20
x3 = 20
y3 = 50

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
potential_energies = np.full((1, 3), (potential_energy_initial_1, potential_energy_initial_2, potential_energy_initial_3))

for loop in range(3000):
    ts = 1
    distance_1_to_2 = distance(x1, y1, x2, y2)
    distance_1_to_3 = distance(x1, y1, x3, y3)
    distance_2_to_3 = distance(x2, y2, x3, y3)

    soften_1_to_2 = soften_f(distance_1_to_2)
    soften_1_to_3 = soften_f(distance_1_to_3)
    soften_2_to_3 = soften_f(distance_2_to_3)

    velocity_x1_increment = (acceleration(m1, m2, m1, distance_1_to_2, soften_1_to_2) * ((x2 - x1) / distance_1_to_2) *
                             ts) + (acceleration(m1, m3, m1, distance_1_to_3, soften_1_to_3) * ((x3 - x1) /
                                                                                                distance_1_to_3) * ts)
    velocity_y1_increment = (acceleration(m1, m2, m1, distance_1_to_2, soften_1_to_2) * ((y2 - y1) / distance_1_to_2) *
                             ts) + (acceleration(m1, m3, m1, distance_1_to_3, soften_1_to_3) * ((y3 - y1) /
                                                                                                distance_1_to_3) * ts)
    velocity_x2_increment = (acceleration(m2, m1, m2, distance_1_to_2, soften_1_to_2) * ((x1 - x2) / distance_1_to_2) *
                             ts) + (acceleration(m2, m3, m2, distance_2_to_3, soften_2_to_3) * ((x3 - x2) /
                                                                                                distance_2_to_3) * ts)
    velocity_y2_increment = (acceleration(m2, m1, m2, distance_1_to_2, soften_1_to_2) * ((y1 - y2) / distance_1_to_2) *
                             ts) + (acceleration(m2, m3, m2, distance_2_to_3, soften_2_to_3) * ((y3 - y2) /
                                                                                                distance_2_to_3) * ts)
    velocity_x3_increment = (acceleration(m3, m1, m3, distance_1_to_3, soften_1_to_3) * ((x1 - x3) / distance_1_to_3) *
                             ts) + (acceleration(m3, m2, m3, distance_2_to_3, soften_2_to_3) * ((x2 - x3) /
                                                                                                distance_2_to_3) * ts)
    velocity_y3_increment = (acceleration(m3, m1, m3, distance_1_to_3, soften_1_to_3) * ((y1 - y3) / distance_1_to_3) *
                             ts) + (acceleration(m3, m2, m3, distance_2_to_3, soften_2_to_3) * ((y2 - y3) /
                                                                                                distance_2_to_3) * ts)

    x1 += velocity_x1 * ts
    y1 += velocity_y1 * ts
    x2 += velocity_x2 * ts
    y2 += velocity_y2 * ts
    x3 += velocity_x3 * ts
    y3 += velocity_y3 * ts

    velocity_x1 += velocity_x1_increment
    velocity_y1 += velocity_y1_increment
    velocity_x2 += velocity_x2_increment
    velocity_y2 += velocity_y2_increment
    velocity_x3 += velocity_x3_increment
    velocity_y3 += velocity_y3_increment

    particles_positions_stack = np.full((1, 6), (x1, y1, x2, y2, x3, y3))
    particles_positions = np.vstack([particles_positions, particles_positions_stack])
    particles_velocities_stack = np.full((1, 6), (velocity_x1, velocity_y1, velocity_x2, velocity_y2, velocity_x3,
                                                  velocity_y3))
    particles_velocities = np.vstack([particles_velocities, particles_velocities_stack])

    velocity_1 = math.sqrt((velocity_x1 ** 2) + (velocity_y1 ** 2))
    velocity_2 = math.sqrt((velocity_x2 ** 2) + (velocity_y2 ** 2))
    velocity_3 = math.sqrt((velocity_x3 ** 2) + (velocity_y3 ** 2))
    kinetic_energy_1 = kinetic_energy(m1, velocity_1)
    kinetic_energy_2 = kinetic_energy(m2, velocity_2)
    kinetic_energy_3 = kinetic_energy(m3, velocity_3)
    kinetic_energies_stack = np.full((1, 3), (kinetic_energy_1, kinetic_energy_2, kinetic_energy_3))
    kinetic_energies = np.vstack([kinetic_energies, kinetic_energies_stack])

    distance_1_to_2 = distance(x1, y1, x2, y2)  # Recalculated distance for potential energy
    distance_1_to_3 = distance(x1, y1, x3, y3)
    distance_2_to_3 = distance(x2, y2, x3, y3)

    potential_energy_1 = potential_energy_f(m1, m2, distance_1_to_2) + potential_energy_f(m1, m3, distance_1_to_3)
    potential_energy_2 = potential_energy_f(m2, m1, distance_1_to_2) + potential_energy_f(m2, m3, distance_2_to_3)
    potential_energy_3 = potential_energy_f(m3, m1, distance_1_to_3) + potential_energy_f(m3, m2, distance_2_to_3)
    potential_energies_stack = np.full((1, 3), (potential_energy_1, potential_energy_2, potential_energy_3))
    potential_energies = np.vstack([potential_energies, potential_energies_stack])

    t += ts
    n += 1

    time_list.append(t)

for item in kinetic_energies[:, 0]:
    kinetic_energy_1_sum += item

for item in kinetic_energies[:, 1]:
    kinetic_energy_2_sum += item

for item in kinetic_energies[:, 2]:
    kinetic_energy_3_sum += item

for item in potential_energies[:, 0]:
    potential_energy_1_sum += item

for item in potential_energies[:, 1]:
    potential_energy_2_sum += item

for item in potential_energies[:, 2]:
    potential_energy_3_sum += item

kinetic_energy_average = (kinetic_energy_1_sum + kinetic_energy_2_sum + kinetic_energy_3_sum) / t
potential_energy_average = (potential_energy_1_sum + potential_energy_2_sum + potential_energy_3_sum) / t

virial_factor = abs(potential_energy_average / kinetic_energy_average)

df_positions = pd.DataFrame(particles_positions, columns=('x1', 'y1', 'x2', 'y2', 'x3', 'y3'))
df_positions.to_csv('Gravitational Dynamics Three Particles (Positions).csv')

df_velocities = pd.DataFrame(particles_velocities, columns=('x1', 'y1', 'x2', 'y2', 'x3', 'y3'))
df_positions.to_csv('Gravitational Dynamics Three Particles (Velocities).csv')

df_kinetic_energy = pd.DataFrame(kinetic_energies, columns=('Particle 1', 'Particle 2', 'Particle 3'))
df_positions.to_csv('Gravitational Dynamics Three Particles (Kinetic Energy).csv')

df_potential_energy = pd.DataFrame(potential_energies, columns=('Particle 1', 'Particle 2', 'Particle 3'))
df_positions.to_csv('Gravitational Dynamics Three Particles (Potential Energy).csv')

plt.plot(particles_positions[:, 0], particles_positions[:, 1], label='Particle 1')
plt.plot(particles_positions[:, 2], particles_positions[:, 3], label='Particle 2')
plt.plot(particles_positions[:, 4], particles_positions[:, 5], label='Particle 3')
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title('Position x vs Position y')
plt.legend()
plt.show()

plt.plot(time_list, particles_positions[:, 0], label='x1')
plt.plot(time_list, particles_positions[:, 1], label='y1')
plt.plot(time_list, particles_positions[:, 2], label='x2')
plt.plot(time_list, particles_positions[:, 3], label='y2')
plt.plot(time_list, particles_positions[:, 4], label='x3')
plt.plot(time_list, particles_positions[:, 5], label='y3')
plt.xlabel('Time (seconds)')
plt.ylabel('Position (meters)')
plt.title('Position vs Time')
plt.legend()
plt.show()

plt.plot(time_list, kinetic_energies[:, 0], label='Particle 1')
plt.plot(time_list, kinetic_energies[:, 1], label='Particle 2')
plt.plot(time_list, kinetic_energies[:, 2], label='Particle 3')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy (Joules)')
plt.title('Kinetic Energy vs Time')
plt.legend()
plt.show()

plt.plot(time_list, potential_energies[:, 0], label='Particle 1')
plt.plot(time_list, potential_energies[:, 1], label='Particle 2')
plt.plot(time_list, potential_energies[:, 2], label='Particle 3')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy (Joules)')
plt.title('Potential Energy vs Time')
plt.legend()
plt.show()

print(virial_factor)
