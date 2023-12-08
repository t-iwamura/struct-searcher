import re
from pathlib import Path
from typing import List

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
