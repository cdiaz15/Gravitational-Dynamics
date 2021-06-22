import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

m1 = 200000000
m2 = 200000000

x1_initial = 0
y1_initial = 0
x2_initial = 20
y2_initial = 0

velocity_x1_initial = 0
velocity_y1_initial = 0
velocity_x2_initial = 0
velocity_y2_initial = 0

t = 0
n = 0

coefficients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
time_steps = [0.01, 0.1, 1]


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def acceleration(m1, m2, m, x, soften):
    return ((6.67408e-11 * m1 * m2) / (x ** 2 + soften)) / m


def kinetic_energy(m, v):
    return 0.5 * m * (v ** 2)


def potential_energy_f(m1, m2, x):
    return (- 6.67408e-11 * m1 * m2) / x


def initial_mass_center(m1, m2, x1_initial, x2_initial, y1_initial,
                        y2_initial):
    mass_center_x = ((m1 * x1_initial) + (m2 * x2_initial)) / (m1 + m2)
    mass_center_y = ((m1 * y1_initial) + (m2 * y2_initial)) / (m1 + m2)
    return mass_center_x, mass_center_y


def virial_moment(m1, m2, mass_center_x, mass_center_y, x, xn, yn):
    moment_arm = math.sqrt((mass_center_x - xn) ** 2 + (mass_center_y - yn) ** 2)
    return ((- 6.67408e-11 * m1 * m2)/ (x ** 2)) * moment_arm
    

props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)  
# Props for bbox parameter in the text box of the graphs. Required to create 
# the soften and potentials labels in the plots

initial_virial_coefficients_array = True

x = distance(x1_initial, y1_initial, x2_initial, y2_initial)
mass_center_x, mass_center_y = initial_mass_center(m1, m2, x1_initial, 
                                                   x2_initial, y1_initial, 
                                                   y2_initial)

for coefficient in coefficients:
    
    def soften_f(x):
        return math.exp(- coefficient * x)
    
    initial_virial_factor_array = True
    initial_virial_moment_average = True
    
    for time_step in time_steps:
        
        iterations = int(1500 / time_step) # calculating the requiered amount of 
        # iterations for the desired amount of time
        
        x1 = x1_initial
        y1 = y1_initial
        x2 = x2_initial
        y2 = y2_initial
        
        velocity_x1 = velocity_x1_initial
        velocity_y1 = velocity_y1_initial
        velocity_x2 = velocity_x2_initial
        velocity_y2 = velocity_y2_initial
        
        time_list = [t]
        
        particles_positions = np.full((1, 4), (x1, y1, x2, y2))
        particles_velocities = np.full((1, 4), (velocity_x1, velocity_y1, 
                                                velocity_x2, velocity_y2))
        
        kinetic_energy_sum_1 = 0
        kinetic_energy_sum_2 = 0
        potential_energy_sum = 0
        virial_moment_sum_1 = 0
        virial_moment_sum_2 = 0
        
        velocity_1 = math.sqrt((velocity_x1 ** 2) + (velocity_y1 ** 2))
        velocity_2 = math.sqrt((velocity_x2 ** 2) + (velocity_y2 ** 2))
    
        kinetic_energy_initial_1 = kinetic_energy(m1, velocity_1)
        kinetic_energy_initial_2 = kinetic_energy(m2, velocity_2)
        kinetic_energies = np.full((1, 2), (kinetic_energy_initial_1, 
                                            kinetic_energy_initial_2))
        
        x = distance(x1, y1, x2, y2)
        
        potential_energy_initial = potential_energy_f(m1, m2, x)
        potential_energies = [potential_energy_initial]
        
        virial_moment_initial_1 = virial_moment(m1, m2, mass_center_x, 
                                                mass_center_y, x, x1, y1)
        virial_moment_initial_2 = virial_moment(m1, m2, mass_center_x, 
                                                mass_center_y, x, x2, y2)
        virial_moments = np.full((1, 2), (virial_moment_initial_1, 
                                          virial_moment_initial_2))
        
        for loop in range(iterations):
            ts = time_step
            x = distance(x1, y1, x2, y2)
            soften = soften_f(x)
        
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
            
            virial_moment_1 = virial_moment(m1, m2, mass_center_x, 
                                            mass_center_y, x, x1, y1)
            virial_moment_2 = virial_moment(m1, m2, mass_center_x, 
                                            mass_center_y, x, x2, y2)
            virial_moments_stack = np.full((1, 2), (virial_moment_1, 
                                                    virial_moment_2))
            virial_moments = np.vstack([virial_moments, virial_moments_stack])
        
            t += ts
            n += 1
        
            time_list.append(t)
        
        for item in kinetic_energies[:, 0]:
            kinetic_energy_sum_1 += item
        
        for item in kinetic_energies[:, 1]:
            kinetic_energy_sum_2 += item
        
        for item in potential_energies:
            potential_energy_sum += item
            
        for item in virial_moments[:, 0]:
            virial_moment_sum_1 += item
        
        for item in virial_moments[:, 1]:
            virial_moment_sum_2 += item
        
        print('Virial Moments')
        print(virial_moment_sum_1)
        print('{} \n'.format(virial_moment_sum_2))
        
        print('Kinetic Energies')
        print(kinetic_energy_sum_1)
        print('{} \n'.format(kinetic_energy_sum_2))
        
        kinetic_energy_average = (kinetic_energy_sum_1 + kinetic_energy_sum_2) / t
        potential_energy_average = ((potential_energy_sum * 2)) / t 
        # Notice that potential energy of both particles is always equal,
        # independently of what the mass of the particles is
        
        virial_moment_average = ((virial_moment_sum_1 + virial_moment_sum_2) / 2) / t
        
        virial_factor = round((virial_moment_average 
                               / kinetic_energy_average), 5)
        
        if initial_virial_factor_array == True:
            virial_factor_array = np.array(virial_factor)
            initial_virial_factor_array = False
        else:
            virial_factor_stack = np.array(virial_factor)
            virial_factor_array = np.vstack([virial_factor_array, 
                                             virial_factor_stack])
        #creates an array to save the virial factor obtained in the simulation 
        #for all the time steps
        
        '''if initial_virial_moment_average == True:
            virial_moment_average_array = np.array(virial_moment_average)
            initial_virial_moment_average = False
        else:
            virial_moment_average_stack = np.array(virial_moment_average)
            virial_moment_average_array = np.vstack([virial_moment_average_array, 
                                             virial_moment_average_stack])
        #creates an array to save the virial moment average obtained in the 
        #simulation for all the time steps'''

        ax_position_time = plt.subplot()
        plt.plot(time_list, particles_positions[:, 0], label='x1')
        plt.plot(time_list, particles_positions[:, 1], label='y1')
        plt.plot(time_list, particles_positions[:, 2], label='x2')
        plt.plot(time_list, particles_positions[:, 3], label='y2')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Position (meters)')
        plt.title('Position vs Time. Coefficient = {}'.format(coefficient))
        ax_position_time.text(0.02, 0.21, 
                             'Virial Factor: \n {}'
                             .format(virial_factor),
                             transform=ax_position_time.transAxes, 
                             verticalalignment='top', bbox=props)
        ax_position_time.text(0.02, 0.076, 'Soften = 'r'$e^{-x*coefficient}$',
                      transform=ax_position_time.transAxes, 
                      verticalalignment='top', bbox=props)
        plt.legend()
        plt.show()
        
        t = 0 # Reset time count for plots and virial calculation
        
        
    if initial_virial_coefficients_array == True:
        virial_factor_array_coefficients = np.array(virial_factor_array)
        initial_virial_coefficients_array = False
    else:
        virial_factor_array_coefficients = np.hstack([virial_factor_array_coefficients, 
                                                   virial_factor_array])
    #saves the virial factors for all the time steps and coeficients (final array)
    
    '''if initial_virial_coefficients_array == True:
        virial_moment_array_coefficients = np.array(virial_moment_average_array)
        initial_virial_coefficients_array = False
    else:
        virial_moment_array_coefficients = np.hstack([virial_moment_array_coefficients, 
                                                   virial_moment_average_array])
    #saves the virial factors for all the time steps and coeficients (final array)'''

df = pd.DataFrame(virial_factor_array_coefficients, index=time_steps, 
                  columns=coefficients)
df.to_csv('Virial Factor Heatmap Data Frame.csv')

heatmap_plot = sns.heatmap(data=df, annot=True, cmap='YlGnBu')
heatmap_plot.set_title('Virial Factor Heatmap.\n Soften = 'r'$e^{-x*coefficient}$. Distance = 20 meters')
heatmap_plot.set_xlabel('Coefficient')
heatmap_plot.set_ylabel('Time Step')


