import random
from math import acos, degrees, sqrt
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure

from struct_searcher.data import load_atom_info
from struct_searcher.fileio import create_lammps_struct_file


def create_niggli_cell(g_max: float, g_min: float = 0.0) -> List[float]:
    """Create Niggli reduced cell

    Args:
        g_max (float): The parameter to control volume maximum.
        g_min (float, optional): The parameter to control volume minimum.
            Defaults to 0.0.

    Returns:
        List[float]: The sequence which represents Niggli reduced cell.
    """
    while True:
        niggli = [random.uniform(g_min, g_max) for _ in range(3)]
        niggli.sort()
        niggli.append(random.uniform(-0.5 * niggli[0], 0.5 * niggli[0]))
        niggli.append(random.uniform(0, 0.5 * niggli[0]))
        niggli.append(random.uniform(0, 0.5 * niggli[1]))

        if (niggli[0] + niggli[1] + 2 * niggli[3]) >= (2 * niggli[4] + 2 * niggli[5]):
            break

    return niggli


def convert_niggli_cell_to_system_params(niggli: List[float]) -> Dict[str, float]:
    """Convert Niggli reduced cell to system parameters

    Args:
        niggli (List[float]): List which represents Niggli reduced cell.

    Returns:
        Dict[str, float]: Dict having system parameters.
    """
    a = sqrt(niggli[0])

    params = {}
    params["xhi"] = a
    params["xy"] = niggli[3] / a
    params["xz"] = niggli[4] / a
    params["yhi"] = sqrt(niggli[1] - params["xy"] ** 2)
    params["yz"] = (niggli[5] - params["xy"] * params["xz"]) / params["yhi"]
    params["zhi"] = sqrt(niggli[2] - params["xz"] ** 2 - params["yz"] ** 2)

    return params


def _calc_angle_from_inner_product(a: float, b: float, val: float) -> float:
    """Calculate angle from inner product

    Args:
        a (float): The norm of first vector.
        b (float): The norm of second vector.
        val (float): The inner product of the vectors.

    Returns:
        float: The angle in units of degree.
    """
    val /= a * b
    if abs(val) > 1:
        angle = 180.0 if val < 0 else 0.0
    else:
        angle = degrees(acos(val))
    return angle


def convert_niggli_cell_to_lattice_constants(niggli: List[float]) -> Tuple[float, ...]:
    """Convert Niggli reduced cell to lattice constants

    Args:
        niggli (List[float]): List which represents Niggli reduced cell.

    Returns:
        Tuple[float, ...]: 6 lattice constants.
    """
    a = sqrt(niggli[0])
    b = sqrt(niggli[1])
    c = sqrt(niggli[2])
    gamma = _calc_angle_from_inner_product(a, b, niggli[3])
    beta = _calc_angle_from_inner_product(c, a, niggli[4])
    alpha = _calc_angle_from_inner_product(b, c, niggli[5])
    return a, b, c, alpha, beta, gamma


def has_enough_space_between_atoms(
    lattice: Lattice,
    frac_coords: NDArray,
    elements: Tuple[str, str],
    n_atom_for_each_element: Tuple[int, int],
) -> bool:
    """Check if a structure has enough space between atoms

    Args:
        lattice (Lattice): Lattice object of a structure.
        frac_coords (NDArray): The fractional coordinates of the atoms.
        elements (Tuple[str, str]): Tuple of element in a structure.
        n_atom_for_each_element (Tuple[int, int]): The number of atoms for each element.

    Returns:
        bool: The result of a check.
    """
    # Create Structure object
    species = [
        elements[i]
        for i, n_atom in enumerate(n_atom_for_each_element)
        for _ in range(n_atom)
    ]
    distances = Structure(lattice, species, frac_coords).distance_matrix

    # Calculate the minimum of atomic distances
    min_distance = 100000
    n_atom = distances.shape[0]
    for i in range(n_atom):
        for j in range(i + 1, n_atom):
            if distances[i, j] < min_distance:
                min_distance = distances[i, j]

    atom_info = load_atom_info()
    d = max(atom_info[e]["distance"] for e in elements)

    return min_distance >= 0.75 * d


def create_sample_struct_file(
    g_max: float, elements: Tuple[str, str], n_atom_for_each_element: Tuple[int, int]
) -> str:
    """Create sample structure file

    Args:
        g_max (float): The parameter to control volume maximum.
        elements (Tuple[str, str]): Tuple of element included in system.
        n_atom_for_each_element (Tuple[int, int]): The number of atoms for each element.

    Returns:
        str: The content of sample structure file.
    """
    cnt = 0
    g_min = 0.0
    while True:
        # Create Niggli reduced cell
        niggli = create_niggli_cell(g_max, g_min)

        # Check that the volume of Niggli cell isn't too large
        a, b, c, alpha, beta, gamma = convert_niggli_cell_to_lattice_constants(niggli)
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        if lattice.volume >= g_max ** (3 / 2):
            continue

        # Create fractional coordinates of atoms
        n_atom = sum(n_atom_for_each_element)
        frac_coords = np.random.rand(n_atom, 3)
        frac_coords[0, :] = 0.0

        if n_atom == 1:
            break

        if has_enough_space_between_atoms(
            lattice, frac_coords, elements, n_atom_for_each_element
        ):
            break
        elif cnt < 1000:
            cnt += 1
            g_min = (cnt / 1000) * g_max

    system_params = convert_niggli_cell_to_system_params(niggli)
    content = create_lammps_struct_file(
        system_params["xhi"],
        system_params["yhi"],
        system_params["zhi"],
        system_params["xy"],
        system_params["xz"],
        system_params["yz"],
        frac_coords,
        elements,
        n_atom_for_each_element,
    )
    return content
