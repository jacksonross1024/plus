# This script shows the uniaxial, prime uniaxial, combined, and cubic
# anisotropy energies for multiple magnetization directions in the xy-plane.

import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Ferromagnet, Grid, World

def polar(uni, cub, uni_prime=None, anisU_dir=(1,0,0), anisU_prime_dir=(0,1,0)):
    zeros = (0, 0, 0)
    # reset all anisotropy parameters
    magnet.ku1, magnet.ku2 = 0, 0
    magnet.anisU = zeros
    magnet.ku1_prime = 0
    magnet.anisU_prime = zeros
    magnet.kc1, magnet.kc2, magnet.kc3 = zeros
    magnet.anisC1 = zeros
    magnet.anisC2 = zeros

    if uni:
        magnet.ku1, magnet.ku2 = uni
        magnet.anisU = anisU_dir

    if uni_prime is not None:
        magnet.ku1_prime = uni_prime
        magnet.anisU_prime = anisU_prime_dir

    if cub:
        magnet.kc1, magnet.kc2, magnet.kc3 = cub
        magnet.anisC1 = (1, 0, 0)
        magnet.anisC2 = (0, 1, 0)

    # precompute normalized directions for theoretical calculation
    if uni:
        u = np.array(anisU_dir, dtype=float)
        u /= np.linalg.norm(u)
    if uni_prime is not None:
        up = np.array(anisU_prime_dir, dtype=float)
        up /= np.linalg.norm(up)

    angles = np.linspace(0, 2 * np.pi, 50)  # sample some magn directions
    energies = []
    energies_theo = []

    for th in angles:
        # set uniform magnetiation in xy-plane for specific direction
        mx, my, mz = np.cos(th), np.sin(th), 0
        magnet.magnetization = (mx, my, mz)
        # save Edens_anisotropy for the only cell
        energies.append(magnet.anisotropy_energy_density.eval()[0,0,0,0])

        # save theoretical anisotropy energy density
        E_theo = 0
        if uni:
            mu = mx * u[0] + my * u[1] + mz * u[2]
            E_theo += -uni[0] * mu**2 - uni[1] * mu**4
        if uni_prime is not None:
            mup = mx * up[0] + my * up[1] + mz * up[2]
            E_theo += -uni_prime * mup**2
        if cub:
            E_theo = cub[0] * (mx**2) * (my**2) + cub[2] * (mx**4) * (my**4)
        energies_theo.append(E_theo)

    # build label and ticks
    if uni and uni_prime is not None:
        s = '-' if uni[0] < 0 else ''
        lab_str = r"$Ku_1 = ${}1, $Ku_1' = ${}0.5 kJ/m³".format(s, s)
        rticks = [-1500, 0, 1500]
    elif uni:
        lab = "Ku_1" if uni[0] else "Ku_2"
        s = '-' if uni[0] < 0 or uni[1] < 0 else ''
        lab_str = r"${} = ${}1 kJ/m³".format(lab, s)
        rticks = [-1000, 0, 1000]
    elif uni_prime is not None:
        s = '-' if uni_prime < 0 else ''
        lab_str = r"$Ku_1' = ${}1 kJ/m³".format(s)
        rticks = [-1000, 0, 1000]
    elif cub:
        lab = "Kc_1" if cub[0] else "Kc_3"
        s = '-' if cub[0] < 0 or cub[2] < 0 else ''
        lab_str = r"${} = ${}1 kJ/m³".format(lab, s)
        rticks = [-250, 0, 250]

    plt.polar(angles, energies_theo, 'k--')
    plt.polar(angles, energies, 'o', label=lab_str)
    ax.set_rticks(rticks)
    ax.set_rlabel_position(0)
    ax.set_xticklabels([])
    plt.legend()


world = World(cellsize=(1e-9, 1e-9, 1e-9))
magnet = Ferromagnet(world, Grid((1, 1, 1)))

fig = plt.figure(figsize=(7, 10.5))
# uniaxial anisotropy
ax = fig.add_subplot(321, projection="polar")
polar([1e3, 0], [])
polar([-1e3, 0], [])
ax = fig.add_subplot(322, projection="polar")
polar([0, 1e3], [])
polar([0, -1e3], [])

# prime uniaxial anisotropy (ku1_prime along (1,1,0))
ax = fig.add_subplot(323, projection="polar")
polar([], [], uni_prime=1e3, anisU_prime_dir=(1, 1, 0))
polar([], [], uni_prime=-1e3, anisU_prime_dir=(1, 1, 0))

# combined uniaxial + prime (ku1 along x, ku1_prime = 0.5*ku1 along y)
ax = fig.add_subplot(324, projection="polar")
polar([1e3, 0], [], uni_prime=0.5e3, anisU_prime_dir=(0, 1, 0))
polar([-1e3, 0], [], uni_prime=-0.5e3, anisU_prime_dir=(0, 1, 0))

# cubic anisotropy
ax = fig.add_subplot(325, projection="polar")
polar([], [1e3, 0, 0])
polar([], [-1e3, 0, 0])
ax = fig.add_subplot(326, projection="polar")
polar([], [0, 0, 1e3])
polar([], [0, 0, -1e3])
plt.tight_layout()

plt.show()
