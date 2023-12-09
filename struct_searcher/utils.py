import re
from math import pi
from pathlib import Path
from typing import Dict, List

from pymatgen.io.lammps.data import LammpsData

from struct_searcher.data import load_atom_info
from struct_searcher.fileio import parse_lammps_log
from struct_searcher.struct import has_enough_space_between_atoms

OUTPUTS_DIR_PATH = Path.home() / "struct-searcher" / "data" / "outputs"


def create_n_atom_lists(n_atom: int) -> List[List[int]]:
    """Create n_atom lists

    Args:
        n_atom (int): The number of atoms in unitcell.

    Returns:
        List[List[int]]: List of n_atom_for_each_type.
    """
    n_atom_lists = [[n, n_atom - n] for n in range(n_atom + 1)]
    return n_atom_lists


def create_formula_dir_path(
    elements: List[str], n_atom_for_each_element: List[int]
) -> Path:
    """Create Path object of formula directory

    Args:
        elements (List[str]): The elements included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.

    Returns:
        Path: Path object of formula directory.
    """
    system_name = "-".join(elements)
    n_atom_id = str(sum(n_atom_for_each_element)).zfill(2)
    formula = "-".join(f"{e}{n}" for e, n in zip(elements, n_atom_for_each_element))
    formula_dir_path = (
        OUTPUTS_DIR_PATH / system_name / "csp" / f"n_atom_{n_atom_id}" / formula
    )
    return formula_dir_path


def calc_begin_id_of_dir(root_dir_path: Path, n_digit: int) -> int:
    """Calculate begin ID of a child directory

    Args:
        root_dir_path (Path): Object of root directory.
        n_digit (int): The number of digits in ID.

    Returns:
        int: The begin ID of a child directory.
    """
    existing_ids = sorted(
        int(p.name)
        for p in root_dir_path.glob("*")
        if p.is_dir() and re.search(rf".*\/\d{{{n_digit}}}", str(p))
    )
    if len(existing_ids) == 0:
        begin_id = 1
    else:
        begin_id = existing_ids[-1] + 1

    return begin_id


def check_previous_relaxation(
    calc_stats: Dict[str, float], output_dir_path: Path, relaxation_id: str
) -> str:
    """Check previous relaxation

    Args:
        calc_stats (Dict[str, float]): Dict about calculation results.
        output_dir_path (Path): Object of output directory.
        relaxation_id (str): The ID of relaxation.

    Returns:
        str: The result status of LAMMPS calculation.
    """
    result_status = parse_lammps_log(
        str(output_dir_path / f"log_{relaxation_id}.lammps")
    )
    if calc_stats["energy"] >= 1e08 or calc_stats["energy"] <= -1e03:
        result_status = "stop"

    # Check volume
    structure = LammpsData.from_file(
        str(output_dir_path / f"final_structure_{relaxation_id}"), atom_style="atomic"
    ).structure
    species = [specie.symbol for specie in structure.species]
    elements = list(set(species))
    atom_info = load_atom_info()
    d = max(atom_info[e]["distance"] for e in elements)
    max_volume = 200 * 4 * pi * (0.5 * d) ** 3 / 3

    if structure.volume >= max_volume:
        result_status = "stop"

    # Check nearest neighbor distance
    if result_status != "stop":
        n_atom_for_each_element = [species.count(e) for e in elements]
        if not has_enough_space_between_atoms(
            structure.lattice,
            structure.frac_coords,
            elements,
            n_atom_for_each_element,
            dtol=1e-02,
        ):
            result_status = "stop"

    return result_status
