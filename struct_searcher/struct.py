import random
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np

from struct_searcher.fileio import create_lammps_struct_file


def create_niggli_cell(g_max: float) -> List[float]:
    """Create Niggli reduced cell

    Args:
        g_max (float): The parameter, g_max.

    Returns:
        List[float]: The sequence which represents Niggli reduced cell.
    """
    while True:
        niggli = [random.uniform(0, g_max) for _ in range(3)]
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


def create_sample_struct_file(
    g_max: float, n_atom_for_each_element: Tuple[int, int]
) -> str:
    """Create sample structure file

    Args:
        g_max (float): The parameter, G_max.
        n_atom_for_each_element (Tuple[int, int]): The number of atoms for each element.

    Returns:
        str: The content of sample structure file.
    """
    # Create Niggli reduced cell
    niggli = create_niggli_cell(g_max)
    system_params = convert_niggli_cell_to_system_params(niggli)

    # Create fractional coordinates of atoms
    n_atom = sum(n_atom_for_each_element)
    frac_coords = np.random.rand(n_atom, 3)
    frac_coords[0, :] = 0.0

    content = create_lammps_struct_file(
        system_params["xhi"],
        system_params["yhi"],
        system_params["zhi"],
        system_params["xy"],
        system_params["xz"],
        system_params["yz"],
        frac_coords,
        n_atom_for_each_element,
    )
    return content
