import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

m1 = 200000000
m2 = 200000000

x1 = 0
y1 = 0
x2 = 10
y2 = 0

velocity_x1 = 0
velocity_y1 = 0
velocity_x2 = 0
velocity_y2 = 0

x1_initial = x1
y1_initial = y1
x2_initial = x2
y2_initial = y2

velocity_x1_initial = velocity_x1
velocity_y1_initial = velocity_y1
velocity_x2_initial = velocity_x2
velocity_y2_initial = velocity_y2

particles_positions = np.full((1, 4), (x1, y1, x2, y2))
particles_velocities = np.full((1, 4), (velocity_x1, velocity_y1, velocity_x2, 
                                        velocity_y2))

t = 0
n = 0

kinetic_energy_1_sum = 0
kinetic_energy_2_sum = 0
potential_energy_sum = 0

time_list = [t]


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def acceleration(m1, m2, m, x, soften):
    return ((6.67408e-11 * m1 * m2) / (x ** 2 + soften)) / m


def kinetic_energy(m, v):
    return 0.5 * m * (v ** 2)


def potential_energy_f(m1, m2, x):
    return (- 6.67408e-11 * m1 * m2) / x


def soften_f(x):
    return math.exp(- x/6)


velocity_1 = math.sqrt((velocity_x1 ** 2) + (velocity_y1 ** 2))
velocity_2 = math.sqrt((velocity_x2 ** 2) + (velocity_y2 ** 2))

kinetic_energy_initial_1 = kinetic_energy(m1, velocity_1)
kinetic_energy_initial_2 = kinetic_energy(m2, velocity_2)
kinetic_energies = np.full((1, 2), (kinetic_energy_initial_1, kinetic_energy_initial_2))

x = distance(x1, y1, x2, y2)

potential_energy_initial = potential_energy_f(m1, m2, x)
potential_energies = [potential_energy_initial]

soften_array = np.full((1, 1), soften_f(x))

for loop in range(30000):
    ts = 0.1
    x = distance(x1, y1, x2, y2)
    soften = soften_f(x)
    soften_array_stack = np.full((1, 1), soften)
    soften_array = np.vstack([soften_array, soften_array_stack])

    velocity_x1_increment = acceleration(m1, m2, m1, x, soften) * ((x2 - x1) / x) * ts
    velocity_y1_increment = acceleration(m1, m2, m1, x, soften) * ((y2 - y1) / x) * ts
    velocity_x2_increment = acceleration(m1, m2, m2, x, soften) * ((x1 - x2) / x) * ts
    velocity_y2_increment = acceleration(m1, m2, m2, x, soften) * ((y1 - y2) / x) * ts

    x1 += velocity_x1 * ts
    y1 += velocity_y1 * ts
    x2 += velocity_x2 * ts
    y2 += velocity_y2 * ts

    velocity_x1 += velocity_x1_increment
    velocity_y1 += velocity_y1_increment
    velocity_x2 += velocity_x2_increment
    velocity_y2 += velocity_y2_increment

    particles_positions_stack = np.full((1, 4), (x1, y1, x2, y2))
    particles_positions = np.vstack([particles_positions, particles_positions_stack])
    particles_velocities_stack = np.full((1, 4), (velocity_x1, velocity_y1, velocity_x2, velocity_y2))
    particles_velocities = np.vstack([particles_velocities, particles_velocities_stack])

    velocity_1 = math.sqrt((velocity_x1 ** 2) + (velocity_y1 ** 2))
    velocity_2 = math.sqrt((velocity_x2 ** 2) + (velocity_y2 ** 2))
    kinetic_energy_1 = kinetic_energy(m1, velocity_1)
    kinetic_energy_2 = kinetic_energy(m2, velocity_2)
    kinetic_energies_stack = np.full((1, 2), (kinetic_energy_1, kinetic_energy_2))
    kinetic_energies = np.vstack([kinetic_energies, kinetic_energies_stack])

    x = distance(x1, y1, x2, y2)  # Recalculated distance for potential energy
    potential_energy = potential_energy_f(m1, m2, x)
    potential_energies.append(potential_energy)

    t += ts
    n += 1

    time_list.append(t)

for item in kinetic_energies[:, 0]:
    kinetic_energy_1_sum += item

for item in kinetic_energies[:, 1]:
    kinetic_energy_2_sum += item

for item in potential_energies:
    potential_energy_sum += item

kinetic_energy_average = (kinetic_energy_1_sum + kinetic_energy_2_sum) / t
potential_energy_average = (potential_energy_sum * 2) / t  # Notice that potential energy of both particles is
# always equal

virial_factor = round((potential_energy_average / kinetic_energy_average), 5)

df_positions = pd.DataFrame(particles_positions, columns=('x1', 'y1', 'x2', 'y2'))
df_positions.to_csv('Gravitational Dynamics Two Particles (Positions).csv')

df_velocities = pd.DataFrame(particles_velocities, columns=('x1', 'y1', 'x2', 'y2'))
df_positions.to_csv('Gravitational Dynamics Two Particles (Velocities).csv')

df_kinetic_energy = pd.DataFrame(kinetic_energies, columns=('Particle 1', 'Particle 2'))
df_positions.to_csv('Gravitational Dynamics Two Particles (Kinetic Energy).csv')

df_potential_energy = pd.DataFrame(potential_energies)
df_positions.to_csv('Gravitational Dynamics Two Particles (Potential Energy).csv')

df_soften = pd.DataFrame(soften_array)
df_soften.to_csv('Gravitational Dynamics Soften Value.csv')

df_time = pd.DataFrame(time_list)
df_time.to_csv('Gravitational Dynamics Time List.csv')

props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)  # Props for bbox parameter in the text
# box of the graphs

'''ax_position = plt.subplot()
ax_position.plot(particles_positions[:, 0], particles_positions[:, 1], label='Particle 1')
ax_position.plot(particles_positions[:, 2], particles_positions[:, 3], label='Particle 2')
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title('Position x vs Position y')
ax_position.text(0.02, 0.25, 'Virial Factor: \n     {}'.format(virial_factor), transform=ax_position.transAxes,
                 verticalalignment='top', bbox=props)
ax_position.text(0.02, 0.01, 'Soften = 'r'$e^{-x}$',
                 transform=ax_position.transAxes, verticalalignment='top', bbox=props)
ax_position.text(0.875, 0.97, ' x1 = {}\n y1 = {}\n x2 = {}\n y2 = {}'.format(x1_initial, y1_initial, x2_initial, y2_initial
                                                                          ), transform=ax_position.transAxes,
                 verticalalignment='top', bbox=props)
ax_position.text(0.705, 0.97, 'vx1 = {}\n vy1 = {}\n vx2 = {}\n vy2 = {}'.format(velocity_x1_initial, velocity_y1_initial,
                                                                              velocity_x2_initial, velocity_y2_initial),
                 transform=ax_position.transAxes, verticalalignment='top', bbox=props)
ax_position.text(0.716, 0.77, 'Mass 1 = {}\nMass 2 = {}'.format(m1, m2), transform=ax_position.transAxes,
                 verticalalignment='top', bbox=props)
plt.legend()
plt.show()'''

ax_position_time = plt.subplot()
plt.plot(time_list, particles_positions[:, 0], label='x1')
plt.plot(time_list, particles_positions[:, 1], label='y1')
plt.plot(time_list, particles_positions[:, 2], label='x2')
plt.plot(time_list, particles_positions[:, 3], label='y2')
plt.xlabel('Time (seconds)')
plt.ylabel('Position (meters)')
plt.title('Position vs Time')
#ax_position_time.text(0.02, 0.21, 'Virial Factor: \n      {}'.format(virial_factor),
                    #  transform=ax_position_time.transAxes, verticalalignment='top', bbox=props)
#ax_position_time.text(0.02, 0.076, 'Soften = 'r'$e^{-x/2}$',
                    #  transform=ax_position_time.transAxes, verticalalignment='top', bbox=props)
plt.legend()
plt.show()

ax_kinetic_energy = plt.subplot()
plt.plot(time_list, kinetic_energies[:, 0], label='Particle 1')
plt.plot(time_list, kinetic_energies[:, 1], label='Particle 2')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy (Joules)')
plt.title('Kinetic Energy vs Time')
#ax_kinetic_energy.text(0.02, 0.175, 'Virial Factor: \n      {}'.format(virial_factor),
                     #  transform=ax_kinetic_energy.transAxes, verticalalignment='top', bbox=props)
#ax_position_time.text(0.02, 0.065, 'Soften = 'r'$e^{-x/6}$',
                     # transform=ax_kinetic_energy.transAxes, verticalalignment='top', bbox=props)
plt.legend()
plt.show()

ax_velocity_x = plt.subplot()
plt.plot(time_list, particles_velocities[:, 0], label='Particle 1')
plt.plot(time_list, particles_velocities[:, 2], label='Particle 2')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
#ax_kinetic_energy.text(0.02, 0.175, 'Virial Factor: \n      {}'.format(virial_factor),
                      # transform=ax_kinetic_energy.transAxes, verticalalignment='top', bbox=props)
#ax_position_time.text(0.02, 0.065, 'Soften = 'r'$e^{-x/6}$',
                    #  transform=ax_kinetic_energy.transAxes, verticalalignment='top', bbox=props)
#plt.legend()
plt.show()

'''ax_potential_energy = plt.subplot()
plt.plot(time_list, potential_energies, label='Particle 1')
plt.plot(time_list, potential_energies, label='Particle 2')
plt.xlabel('Time (seconds)')
plt.ylabel('Energy (Joules)')
plt.title('Potential Energy vs Time')
ax_potential_energy.text(0.02, 0.175, 'Virial Factor: \n      {}'.format(virial_factor),
                         transform=ax_potential_energy.transAxes, verticalalignment='top', bbox=props)
ax_potential_energy.text(0.02, 0.065, 'Soften = 'r'$e^{-x/6}$',
                         transform=ax_potential_energy.transAxes, verticalalignment='top', bbox=props)
plt.legend()
plt.show()'''

'''ax_soften = plt.subplot()
plt.plot(time_list, soften_array)
plt.xlabel('Time (seconds)')
plt.ylabel('Soften value')
plt.title('Soften vs Time')
ax_soften.text(0.02, 0.175, 'Virial Factor: \n      {}'.format(virial_factor),
                         transform=ax_soften.transAxes, verticalalignment='top', bbox=props)
ax_soften.text(0.02, 0.065, 'Soften = 'r'$e^{-x/6}$ + 'r'$(1/2)*e^{-x}$',
                         transform=ax_soften.transAxes, verticalalignment='top', bbox=props)
plt.show()'''
