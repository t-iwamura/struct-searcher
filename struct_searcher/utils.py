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
        OUTPUTS_DIR_PATH / system_name / "multi-start" / f"n_atom_{n_atom_id}" / formula
    )
    return formula_dir_path
