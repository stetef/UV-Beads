import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import itertools


def normal(x, mu, sig):
    E = np.exp( -0.5 * ((x - mu) / sig)**2 )
    return 4 * ( 1 / sig * np.sqrt(2 * np.pi) ) * E


def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.2
    
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, A)


def get_color_spectrum_from_indices(indices, colors, spectra):
    new_color = np.average([colors[idx] for idx in indices if idx != -1], axis=0)
    new_spectrum = np.sum([spectra[idx] for idx in indices if idx != -1], axis=0)
    return new_color, new_spectrum


def show_lattice(plot, indices, atom_colors):
    fig, axes = plot
    alphas = [1 if idx != -1 else 0 for idx in indices]
    for i, ax in enumerate(axes):
        if alphas[i] != 0:
            r = 0.2 + 0.1 * (indices[i] - 1)
            circle = plt.Circle((0.5, 0.5), r, edgecolor='w',
                                facecolor=atom_colors[indices[i] - 1])
            ax.add_patch(circle)
        ax.axvline(0.5, c=plt.cm.tab10(7), linewidth=4, alpha=0.8, zorder=0)
        ax.axhline(0.5, c=plt.cm.tab10(7), linewidth=4, alpha=0.8, zorder=0)
        ax.axis('off')

        
def bond_length_to_wavelength(bond_length):
    return bond_length * 100


def get_list_subsets_by_ele(sublists, idx_list, ele):
    try:
        split = idx_list.index(ele)
        sublists.append(idx_list[:split])
        get_list_subsets_by_ele(sublists, idx_list[split + 1:], ele)
        return sublists
    except ValueError:
        sublists.append(idx_list)
        return

    
def get_bonds(sublists):
    bond_list = []
    for sublist in sublists:
        n = len(sublist)
        bonds = np.sort([(sublist[i], sublist[i + 1])
                         for i in range(n - 1)], axis=0)
        bond_list.append(bonds)
    return bond_list


def get_longer_resonances(resonances, length_list):
    n = len(length_list)
    longer_res = np.sort([(length_list[i], length_list[i + 1])
                          for i in range(n - 1)], axis=0)
    total_lengths = np.sum(longer_res, axis=1)
    if len(total_lengths) == 1:
        resonances.append(total_lengths[0])
    else:
        for length in total_lengths:
            resonances.append(length) 
        get_longer_resonances(resonances, total_lengths)
    return resonances


def get_resonances_from_bonds(resonances, bonds, bond_lengths):
    for pair in bonds:
        bond_length = bond_lengths[tuple(pair)]
        resonances.append(bond_length)
    if len(resonances) == 1:
        return(resonances)
    else:
        return get_longer_resonances(resonances, resonances)
    

def show_example_lattice(mylattice, atom_colors, bond_lengths=None,
                         plot_encode=False, plot_resonances=False):
    ncols = len(mylattice)
    fig, axes = plt.subplots(figsize=(ncols * 1.5, 1.5), ncols=ncols)
    plt.subplots_adjust(wspace=0)
    show_lattice((fig, axes), mylattice, atom_colors)
    for i, idx in enumerate(mylattice):
        if idx != -1:
            axes[i].set_title(f'Atom {idx}', fontsize=16)
    if bond_lengths is not None and plot_resonances is False:
        for i, idx in enumerate([0, 3, 6]):
            label = f'\nBond length\n= {list(bond_lengths.values())[i]} $\AA$'
            axes[idx].text(1, 0, label, ha='center', va='top',
                            transform=axes[idx].transAxes, fontsize=16)
    if plot_encode:
        for i, idx in enumerate(mylattice):
            axes[i].text(0.5, 0, f'\n{idx}', ha='center', va='top',
                         transform=axes[i].transAxes, fontsize=20)
    if bond_lengths is not None and plot_resonances:
        for i, idx in enumerate([0, 1, 4, 5]):
            label = f'{list(bond_lengths.values())[mylattice[idx]]} $\AA$'
            axes[idx].text(1, 0, label, ha='center', va='top',
                           transform=axes[idx].transAxes, fontsize=16)
        for i, idx in enumerate([1, 5]):
            label = f'\n|____________________|\n' + \
                    f'{list(bond_lengths.values())[mylattice[idx]] * 2} $\AA$'
            axes[idx].text(0.5, 0, label, ha='center', va='top',
                           transform=axes[idx].transAxes, fontsize=16)
            

def get_color_spectrum_from_lattice(mylattice, bond_lengths, atom_colors):
    # check for correct encoding
    for e in mylattice:
        if e not in [-1, 1, 2]:
            print(f'{e} is not a 1, 2, or -1. Try again!')
            return
    # check if more than one nanoparticle
    if -1 in mylattice:
        if mylattice[0] == -1 and mylattice[-1] == -1:
            print('Sandwich!')
            mylattice = mylattice[1:-1]
        elif mylattice.index(-1) == 0:
            mylattice = mylattice[1:]
        elif mylattice.index(-1) == len(mylattice) - 1:
            mylattice = mylattice[:-1]
        if -1 in mylattice:
            sublists = get_list_subsets_by_ele([], mylattice, -1)
        else:
            sublists = [mylattice]
    else:
        sublists = [mylattice]
    # remove chains if only one atom long
    filtered_sublists = []
    for subchain in sublists:
        if len(subchain) > 1:
            filtered_sublists.append(subchain)
    # get all bonds from atom chains
    bond_list = get_bonds(filtered_sublists)
    # get wavelenghts
    all_wavelengths = []
    for bonds in bond_list:
        resonances = get_resonances_from_bonds([], bonds, bond_lengths)
        wavelengths = [bond_length_to_wavelength(length)
                       for length in resonances]
        for wavelength in wavelengths:
            all_wavelengths.append(wavelength)
    visible_wavelengths = [wavelength for wavelength in all_wavelengths
                           if wavelength > 380 and wavelength < 750]
    
    print('Your lattice has the following wavelengths in the visible ' + 
          f'light spectrum:\n{[int(wv) for wv in visible_wavelengths]} (in nm)')
    colors = [wavelength_to_rgb(wavelength)
              for wavelength in visible_wavelengths]
    visible_light = np.linspace(300, 830, 1000)
    spectra = [normal(visible_light, wavelength, 10)
               for wavelength in visible_wavelengths]

    print('\nThose wavelengths look like:')
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, spectrum in enumerate(spectra):
        ax.plot(visible_light, spectrum, color=colors[i])
    ax.tick_params(length=6, width=1.5, labelsize=14)
    ax.set_xlabel('Wavelength [nm]', fontsize=16)
    plt.show()
    
    print('\nWith all resonances, the total visible light spectrum' +
          'from your lattice looks like this:')
    if len(colors) != 0:
        new_color, new_spectrum = get_color_spectrum_from_indices(
            np.arange(len(colors)),
            colors, spectra)
    else:
        new_color = 'k'
        new_spectrum = np.zeros(len(visible_light))
    ncols = len(mylattice)
    fig = plt.figure(figsize=(1. * ncols, 5))
    spec = fig.add_gridspec(nrows=2, ncols=ncols,
                            height_ratios=[0.3, 1])
    plt.subplots_adjust(hspace=0.3, wspace=0)
    axes = [fig.add_subplot(spec[0, i]) for i in range(len(mylattice))]
    axes.append(fig.add_subplot(spec[1, :]))
    
    show_lattice((fig, axes[:-1]), mylattice, atom_colors)
    ax = axes[-1]
    ax.plot(visible_light, new_spectrum, color=new_color, linewidth=3)
    ax.tick_params(length=6, width=1.5, labelsize=14)
    ax.set_xlabel('Wavelength [nm]', fontsize=16)
    ax.set_ylabel('Intensity\n(arb. units)', fontsize=20)
    plt.show()