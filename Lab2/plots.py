import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

def get_x_axis(a_0, num_atoms, stepsize):
    return np.arange(start=-(num_atoms) * a_0 / 2, 
                     stop=(num_atoms) * a_0 / 2, 
                     step=stepsize)

def get_monoatomic_electron_density(a_0, num_atoms, stepsize, 
                                    amplitude, sigma, epsilon=0.0,
                                    thermal_vibrations=False):
    x_grid = get_x_axis(a_0, num_atoms, stepsize)
    electron_density = np.zeros_like(x_grid)

    if thermal_vibrations:
        atom_positions = -(num_atoms - 1) * a_0 / 2 + np.arange(num_atoms) * a_0 \
                         + epsilon * (2 * np.random.random(num_atoms) - 1)
    else:
        atom_positions = -(num_atoms - 1) * a_0 / 2 + np.arange(num_atoms) * a_0

    for atom_position in atom_positions:
        current_atom_electron_density = amplitude * \
            np.exp(-(x_grid - atom_position)**2 / (2 * sigma**2))
        electron_density += current_atom_electron_density

    return x_grid, electron_density

def get_diatomic_electron_density(a_0,num_atoms, stepsize, amplitude_1,
                                  amplitude_2, sigma_1, sigma_2, epsilon=0.0,
                                  thermal_vibrations=False):

    x_grid = get_x_axis(a_0,num_atoms,stepsize)

    electron_density = np.zeros_like(x_grid)

    # Generate atom positions
    if thermal_vibrations:
        atom_positions = -(num_atoms - 1) * a_0 / 2 + np.arange(num_atoms) * a_0 \
                         + epsilon * (2 * np.random.random(num_atoms) - 1)
    else:
        atom_positions = -(num_atoms - 1) * a_0 / 2 + np.arange(num_atoms) * a_0

    # Loop over each atom, alternating between amplitude_1/sigma_1 and amplitude_2/sigma_2
    for i, atom_position in enumerate(atom_positions):
        if i % 2 == 0:  # Even index: use amplitude_2 and sigma_2
            current_atom_electron_density = amplitude_2 * \
                np.exp(-(x_grid - atom_position)**2 / (2 * sigma_2**2))
        else:  # Odd index: use amplitude_1 and sigma_1
            current_atom_electron_density = amplitude_1 * \
                np.exp(-(x_grid - atom_position)**2 / (2 * sigma_1**2))
        
        electron_density += current_atom_electron_density
    
    return x_grid, electron_density


def plot_diffraction_intensity(electron_density, stepsize, tukey_window=True, alpha=0.5):
     # Apply Tukey window if enabled
    if tukey_window:
        tukey_win = tukey(len(electron_density), alpha=alpha)
        electron_density = electron_density * tukey_win
    
    # Compute the Fourier Transform of the electron density
    fft_density = np.fft.fftshift(np.fft.fft(electron_density))
    
    # Calculate the intensity (square of the magnitude of the FFT)
    intensity = np.abs(fft_density)**2

    # Create the Q-axis (reciprocal space axis)
    q_axis = (np.arange(len(fft_density)) - len(fft_density) / 2) /\
             (len(fft_density) / 2 * stepsize * np.pi)
    
    return q_axis, intensity

def get_glass_electron_density(a_0, num_atoms, stepsize, amplitude_1, 
                               sigma_1, epsilon_glass):
    # Create the spatial grid
    x_grid = get_x_axis(a_0, num_atoms, stepsize)

    # Generate initial atom positions (without disorder)
    atom_positions = -(num_atoms - 1) * a_0 / 2 + np.arange(num_atoms) * a_0

    # Apply random displacements for each atom (epsilon_glass)
    random_displacements = epsilon_glass * (2 * np.random.random(num_atoms) - 1)
    atom_positions += random_displacements

    # Compute the electron density for all atoms in a vectorized way
    electron_density = amplitude_1 * np.exp(-(x_grid[:, np.newaxis] - atom_positions)**2 / (2 * sigma_1**2))

    # Sum the contributions of all atoms along the axis corresponding to atoms
    total_electron_density = np.sum(electron_density, axis=1)

    return x_grid, total_electron_density



if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(80085)
    # Parameters
    A_0=2.
    NUM_ATOMS=100
    STEPSIZE=0.001
    EPSILON=1.2
    EPSILON_GLASS = 0.5
    THERMAL = True

    # Different amplitudes and widths to test
    # amplitude_values_1 = [1.0, 2.0, 3.0]  # Different amplitudes
    amplitude_values_1 = [1.0, 1.0, 1.0]  # Different amplitudes
    sigma_values_1 = [0.1, 0.2, 0.3]     # Different Gaussian widths

    # amplitude_values_2 = [1.5, 2.5, 3.5]  # Different amplitudes
    amplitude_values_2 = [2., 2., 2.]  # Different amplitudes
    sigma_values_2 = [0.2, 0.3, 0.4]     # Different Gaussian widths

    # Create figure for side-by-side plots
    fig, axs = plt.subplots(len(amplitude_values_1), 2, figsize=(12, 10))
    
    # Plot for diatomic model
    def plot_diatomic()->None:
        for i, (amplitude1, sigma1, amplitude2, sigma2) in enumerate(zip(amplitude_values_1, sigma_values_1, 
                                                                         amplitude_values_2, sigma_values_2)):
            x_grid, electron_density = get_diatomic_electron_density(A_0, NUM_ATOMS, STEPSIZE, amplitude1,amplitude2,
                                                                     sigma1, sigma2, EPSILON, THERMAL)
            freq, intensity = plot_diffraction_intensity(electron_density, STEPSIZE)

            # Plot Electron Density (left column)
            axs[i, 0].plot(x_grid, electron_density, label=f'$A_n1$ =  {amplitude1}, $\\sigma_n1$ = {sigma1} \n $A_n2$ = {amplitude2}, $\\sigma_n2={sigma2}$', color='blue')
            axs[i, 0].set_xlabel('Distance [Å]')
            axs[i, 0].set_ylabel('Electron Density [arb.]')
            axs[i, 0].legend(loc=1)

            # Plot Diffraction Intensity (right column)
            axs[i, 1].plot(freq, intensity, label='Intensity', color='red')
            axs[i, 1].set_xlabel('Frequency [1/Å]')
            axs[i, 1].set_ylabel('Intensity')
            axs[i, 1].legend()

        plt.tight_layout()
        # plt.savefig('Figures/dia_thermal.png')
        plt.show()


    def plot_mono() -> None:
        for i, (amplitude, sigma) in enumerate(zip(amplitude_values_1, sigma_values_1)):

            x_grid, electron_density = get_monoatomic_electron_density(
                A_0, NUM_ATOMS, STEPSIZE, amplitude, sigma, EPSILON, THERMAL)
            freq, intensity = plot_diffraction_intensity(electron_density, STEPSIZE)

             # Plot Electron Density (left column)
            axs[i, 0].plot(x_grid, electron_density, label=f'$A_n$ ={amplitude}, $\\sigma_n$ = {sigma}', color='blue')
            axs[i, 0].set_xlabel('Distance [Å]')
            axs[i, 0].set_ylabel('Electron Density [arb.]')
            axs[i, 0].legend(loc=1)
        
            # Plot Diffraction Intensity (right column)
            axs[i, 1].plot(freq, intensity, label='Intensity', color='red')
            axs[i, 1].set_xlabel('Frequency [1/Å]')
            axs[i, 1].set_ylabel('Intensity')
            axs[i, 1].legend()

        plt.tight_layout()
        plt.savefig('Figures/mono_thermal.png')
        plt.show()

    def plot_glass()->None:
        for i, (amplitude, sigma) in enumerate(zip(amplitude_values_1, sigma_values_1)):
            x_grid, electron_density = get_glass_electron_density(A_0, NUM_ATOMS, STEPSIZE, 
                                                                  amplitude, sigma, EPSILON_GLASS)
            freq, intensity = plot_diffraction_intensity(electron_density, STEPSIZE)

            # Plot Electron Density (left column)

            axs[i, 0].plot(x_grid, electron_density, label=f'$A_n$ ={amplitude}, $\\sigma_n$ = {sigma}', color='blue')
            axs[i, 0].set_xlabel('Distance [Å]')
            axs[i, 0].set_ylabel('Electron Density [arb.]')
            axs[i, 0].legend(loc=1)

            # Plot Diffraction Intensity (right column)
            axs[i, 1].plot(freq, intensity, label='Intensity', color='red')
            axs[i, 1].set_xlabel('Frequency [1/Å]')
            axs[i, 1].set_ylabel('Intensity')
            axs[i, 1].legend()

        plt.tight_layout()
        # plt.savefig('Figures/glass.png')
        plt.show()
    # plot_mono()
    # plot_diatomic()
    plot_glass()

