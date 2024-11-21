# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:37:09 2024

@author: Haraldur Gunnarsson
"""

#Import neccesary modules
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import erf
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

start_time = time.time() #Timing the run

# Constants for diffusion
C0 = 1.0  #Initial concentration in the inclusion (arbitrary units)
D0 = 1e-8  #Pre-exponential factor for diffusion (m^2/s)
Ea = 150e3  #Activation energy for diffusion (J/mol)
R = 8.314  #Universal gas constant (J/(mol*K))
initial_temp = 1200  #Initial temperature (Kelvin)
temp_drop = 150  #Temperature drop over time (Kelvin)

# Simulation parameters
inclusion_radius = 50e-6  #Inclusion radius (meters)
host_size = 0.5e-3  #Host radius (meters)
num_steps = 2000  #Number of spatial steps
dx = host_size / num_steps  #Step size (meters)
x_values = np.linspace(0, host_size, num_steps + 1)  # Array of position values

# Time parameters
total_time = 2e7  #Total simulation time (seconds)
dt = 2e3  #Time step size (seconds)
num_time_steps = int(total_time / dt)  #Total number of time steps

#Observation times for storing profiles
#observation_times = [1e3, 1e4, 1e5, 1e6, 5e6, 7e6, 1e7]  # Observation times in seconds
observation_times = [1e5, 1e6, 5e6, 7e6, 1e7]  # Observation times in seconds

#Function to solve a tridiagonal matrix system using the Thomas algorithm
def solve_tridiagonal(a, b, c, d):
    n = len(d)
    for i in range(1, n): #Fwd elimination
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.zeros(n) #Initialize solution array
    x[-1] = d[-1] / b[-1] #Back subst start at last element
    for i in range(n - 2, -1, -1): #Completing back subst
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x #Return solution array

# Adjusted analytical solution function to incorporate D_effective
def analytical_solution_adjusted(x, t, D0, Ea, R, initial_temp, temp_drop, total_time, C0):
    if t == 0: #Initial condition
        return np.zeros_like(x)
    
    #Compute D_effective by integrating D(t) over time
    #to have better fit of the analytical solution with the numerical solution
    num_integral_steps = 2000  # Resolution of numerical integration
    #dt_eff = t / num_integral_steps  # Small time step for effective calculation
    times_eff = np.linspace(0, t, num_integral_steps + 1)  #Time array for integration
    temps_eff = initial_temp - (temp_drop * times_eff / total_time)  #Temperature at each time step
    D_t = D0 * np.exp(-Ea / (R * temps_eff))  #Diffusion coefficient at each time step
    
    #Integrate D(t) over time to get D_effective
    D_effective = np.trapz(D_t, times_eff) / t  #Numerically integrated average
    #print(D_effective)
    
    #Compute the concentration using the adjusted D_effective
    return C0 * (1 - erf(x / (2 * np.sqrt(D_effective * t))))

#Function to simulate diffusion - Crank-Nicolson method
def run_simulation(num_steps, dx, dt): 
    x_values = np.linspace(0, host_size, num_steps + 1) #Spatial grid
    concentration = np.zeros(num_steps + 1) #Initialize the concentration array
    concentration[x_values <= inclusion_radius] = C0 #Initial inclusion concentration

    profiles = [concentration.copy()] #Store initial concentration profile
    profile_times = [0] #Store initial time

    for step in range(num_time_steps): #Loop timsteps
        current_time = step * dt #Compute current simulation time
        current_temp = initial_temp - (temp_drop * current_time / total_time) #Update temperature
        D = D0 * np.exp(-Ea / (R * current_temp)) #Compute diffusion coefficient
        r = D * dt / dx**2 #Stability parameter

        a = -r / 2 * np.ones(num_steps + 1) #Lower diagonal
        b = (1 + r) * np.ones(num_steps + 1) #Main diagonal
        c = -r / 2 * np.ones(num_steps + 1) #Upper diagonal
        rhs = concentration.copy() #Initialize right hand side

        for i in range(1, num_steps): #The right hand side (RHS) vector
            rhs[i] = (1 - r) * concentration[i] + r / 2 * (concentration[i + 1] + concentration[i - 1])

        b[0], c[0] = 1, 0 #Boundary condition
        b[-1], a[-1] = 1, 0 #Boundary condition

        concentration = solve_tridiagonal(a, b, c, rhs) #Solve system

        if current_time in observation_times: #Save profiles at specified times
            profiles.append(concentration.copy())
            profile_times.append(current_time)

    return x_values, profiles, profile_times #Return the grid

# Run simulations for two resolutions
x_values_coarse, profiles_coarse, times_coarse = run_simulation(num_steps, dx, dt)
x_values_fine, profiles_fine, times_fine = run_simulation(2 * num_steps, dx / 2, dt)

# Richardson extrapolation - not used
#profiles_richardson = []
#for coarse, fine in zip(profiles_coarse, profiles_fine):
#    fine_interp = np.interp(x_values_coarse, x_values_fine, fine)
#    richardson = (4 * fine_interp - coarse) / 3
#    profiles_richardson.append(richardson)

# Adjusted analytical profiles for comparison
adjusted_analytical_profiles = []
for time_analytical in times_coarse: #Loop observation times
    adjusted_analytical_profiles.append(
        analytical_solution_adjusted(
            x_values_coarse, time_analytical, D0, Ea, R, initial_temp, temp_drop, total_time, C0
        )
    ) #Compute the analytical

#2D Plot of concentration profiles
plt.figure(figsize=(10, 6))
for idx, profile in enumerate(profiles_coarse):
    plt.plot(x_values * 1e3, profile, label=f"Numerical t = {times_coarse[idx] / 3600:.1f} hours")
plt.xlabel("Distance from Inclusion Interface (mm)")
plt.ylabel("Fe Concentration (arbitrary units)")
plt.title("Diffusion of Fe in Olivine Host (First 0.5 mm, Numerical)")
plt.legend()
plt.grid()
plt.show()

#Plot the adjusted analytical profiles against numerical and Richardson solutions
plt.figure(figsize=(10, 6))
for idx, profile in enumerate(profiles_coarse):
    plt.plot(x_values * 1e3, profile, label=f"Numerical t = {times_coarse[idx] / 3600:.1f} hours")
    plt.plot(x_values * 1e3, adjusted_analytical_profiles[idx], '--', label=f"Analytical t = {times_coarse[idx] / 3600:.1f} hours")
    #plt.plot(x_values_coarse * 1e3, profiles_richardson[idx], ':', label=f"Richardson t = {times_coarse[idx] / 3600:.1f} hours")
plt.xlabel("Distance from Inclusion Interface (mm)")
plt.ylabel("Fe Concentration (arbitrary units)")
plt.title("Comparison of Numerical, Analytical")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
plt.grid()
plt.show()

#Heatmap of Richardson extrapolation errors
#errors_richardson = [np.abs(rich - ana) for rich, ana in zip(profiles_richardson, adjusted_analytical_profiles)]
X_error, Y_error = np.meshgrid(x_values_coarse * 1e3, np.array(times_coarse) / 3600)
#Z_error = np.array(errors_richardson)
#plt.figure(figsize=(10, 6))
#error_map = plt.contourf(X_error, Y_error, Z_error, levels=50, cmap='inferno')
#plt.colorbar(error_map, label='Error Magnitude (absolute)')
#plt.xlabel('Distance from Inclusion Interface (mm)')
#plt.ylabel('Time (hours)')
#plt.title('Heatmap of Richardson Extrapolation Errors')
#plt.show()

#3D plot of numerical profiles
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x_values_coarse * 1e3, np.array(times_coarse) / 3600)
Z = np.array(profiles_coarse)
ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none")
ax.set_xlabel("Distance from Inclusion Interface (mm)")
ax.set_ylabel("Time (hours)")
ax.set_zlabel("Fe Concentration (arbitrary units)")
ax.set_title("3D Visualization of Fe Diffusion in Olivine Host")
plt.show()

#Contour plot of Fe concentration
X_contour, Y_contour = np.meshgrid(x_values * 1e3, np.array(times_coarse) / 3600)
Z_contour = np.array(adjusted_analytical_profiles)
plt.figure(figsize=(10, 6))
contour = plt.contourf(X_contour, Y_contour, Z_contour, levels=50, cmap='viridis')
plt.colorbar(contour, label='Fe Concentration (arbitrary units)')
plt.xlabel('Distance from Inclusion Interface (mm)')
plt.ylabel('Time (hours)')
plt.title('Contour Plot of Fe Concentration in Olivine Host Over Time')
plt.show()

#Heatmap of numerical errors
numerical_errors = [np.abs(num - ana) for num, ana in zip(profiles_coarse, adjusted_analytical_profiles)]
Z_num_error = np.array(numerical_errors)
#plt.figure(figsize=(10, 6))
#error_map_num = plt.contourf(X_error, Y_error, Z_num_error, levels=50, cmap='inferno')
#plt.colorbar(error_map_num, label='Numerical Error Magnitude (absolute)')
#plt.xlabel('Distance from Inclusion Interface (mm)')
#plt.ylabel('Time (hours)')
#plt.title('Heatmap of Numerical Errors Over Time and Space')
#plt.show()

#Heatmap of analytical diffusion profiles
Z_analytical = np.array(adjusted_analytical_profiles)
plt.figure(figsize=(10, 6))
plt.contourf(X_error, Y_error, Z_analytical, levels=50, cmap='viridis')
plt.colorbar(label='Analytical Fe Concentration (arbitrary units)')
plt.xlabel('Distance from Inclusion Interface (mm)')
plt.ylabel('Time (hours)')
plt.title('Heatmap of Analytical Diffusion Profiles')
plt.show()

#Heatmap of numerical diffusion profiles
Z_numerical = np.array(profiles_coarse)
plt.figure(figsize=(10, 6))
plt.contourf(X_error, Y_error, Z_numerical, levels=50, cmap='plasma')
plt.colorbar(label='Numerical Fe Concentration (arbitrary units)')
plt.xlabel('Distance from Inclusion Interface (mm)')
plt.ylabel('Time (hours)')
plt.title('Heatmap of Numerical Diffusion Profiles')
plt.show()

#Combined heatmap for visual comparison
plt.figure(figsize=(10, 6))
plt.contourf(X_error, Y_error, Z_analytical, levels=50, cmap='viridis', alpha=0.7, label='Analytical')
plt.contourf(X_error, Y_error, Z_numerical, levels=50, cmap='plasma', alpha=0.4, label='Numerical')
plt.colorbar(label='Fe Concentration (arbitrary units)')
plt.xlabel('Distance from Inclusion Interface (mm)')
plt.ylabel('Time (hours)')
plt.title('Combined Heatmap of Analytical and Numerical Diffusion Profiles')
plt.legend(['Analytical', 'Numerical'])
plt.show()

# Define a finer time grid
#fine_time_steps = 500  # Number of finer time steps
#fine_times = np.linspace(0, times_coarse[-1], fine_time_steps)  # Finer time grid

# Interpolate profiles over the finer time grid
#interpolated_profiles = []
#for i in range(len(x_values_coarse)):
#    interp_func = interp1d(times_coarse, [profile[i] for profile in profiles_coarse], kind='linear', fill_value="extrapolate")
#    interpolated_profiles.append(interp_func(fine_times))
#interpolated_profiles = np.array(interpolated_profiles).T  # Shape (fine_time_steps, num_steps + 1)

#Create the heatmap with interpolated profiles
#X_fine, Y_fine = np.meshgrid(x_values_coarse * 1e3, fine_times / 3600)  # Distance in mm, Time in hours
#plt.figure(figsize=(10, 6))
#plt.contourf(X_fine, Y_fine, interpolated_profiles, levels=50, cmap='viridis')
#plt.colorbar(label='Fe Concentration (arbitrary units)')
#plt.xlabel('Distance from Inclusion Interface (mm)')
#plt.ylabel('Time (hours)')
#plt.title('Heatmap of Numerical Diffusion Profiles (Interpolated)')
#plt.show()

#Heatmap of numerical errors
numerical_errors = [np.abs(num - ana) for num, ana in zip(profiles_coarse, adjusted_analytical_profiles)]
Z_num_error = np.array(numerical_errors)
plt.figure(figsize=(10, 6))
error_map_num = plt.contourf(X_error, Y_error, Z_num_error, levels=50, cmap='plasma')
plt.colorbar(error_map_num, label='Numerical Error Magnitude (absolute)')
plt.xlabel('Distance from Inclusion Interface (mm)')
plt.ylabel('Time (hours)')
plt.title('Heatmap of Numerical Errors Over Time and Space')
plt.show()

#Print numerical errors
for idx, (numerical, analytical) in enumerate(zip(profiles_coarse, adjusted_analytical_profiles)):
    error = np.abs(numerical - analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)
    print(f"Time = {times_coarse[idx] / 3600:.1f} hours: Max Error = {max_error:.4e}, Mean Error = {mean_error:.4e}")

end_time = time.time() #Ending the timing of model run
elapsed_time = (end_time - start_time)/60 #Calculate elapsed time
print(f"Elapsed time: {elapsed_time:.2f} minutes") #Printing esapsed time