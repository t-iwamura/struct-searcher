import random
from math import acos, cos, degrees, radians, sqrt
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure

from struct_searcher.data import load_atom_info


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


def _calc_inner_product(a: float, b: float, angle: float) -> float:
    """Calculate inner product

    Args:
        a (float): The norm of first vector.
        b (float): The norm of second vector.
        angle (float): The angle in units of degree.

    Returns:
        float: The inner product of the vectors.
    """
    return a * b * cos(radians(angle))


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


def convert_lattice_constants_to_niggli_cell(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> List[float]:
    """Convert lattice constants to Niggli reduced cell

    Args:
        a (float): The length of 1st lattice vector.
        b (float): The length of 2nd lattice vector.
        c (float): The length of 3rd lattice vector.
        alpha (float): The angle between b and c.
        beta (float): The angle between c and a.
        gamma (float): The angle between a and b.

    Returns:
        List[float]: List which represents Niggli reduced cell.
    """
    niggli = []
    niggli.append(a**2)
    niggli.append(b**2)
    niggli.append(c**2)
    niggli.append(_calc_inner_product(a, b, gamma))
    niggli.append(_calc_inner_product(c, a, beta))
    niggli.append(_calc_inner_product(b, c, alpha))
    return niggli


def has_enough_space_between_atoms(
    lattice: Lattice,
    frac_coords: NDArray,
    elements: List[str],
    n_atom_for_each_element: List[int],
    dtol: Optional[float] = None,
) -> bool:
    """Check if a structure has enough space between atoms

    Args:
        lattice (Lattice): Lattice object of a structure.
        frac_coords (NDArray): The fractional coordinates of the atoms.
        elements (List[str]): List of element in a structure.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        dtol (Optional[float], optional): The tolerance for neighbor distance.
            Defaults to None.

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

    if dtol is None:
        atom_info = load_atom_info()
        dtol = 0.75 * max(atom_info[e]["distance"] for e in elements)

    return min_distance >= dtol
