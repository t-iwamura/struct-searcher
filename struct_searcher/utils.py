from pathlib import Path
from typing import List, Tuple

OUTPUTS_DIR_PATH = Path.home() / "struct-searcher" / "data" / "outputs"


def create_n_atom_tuples(n_atom: int) -> List[Tuple[int, int]]:
    """Create n_atom tuples

    Args:
        n_atom (int): The number of atoms in unitcell.

    Returns:
        List[Tuple[int, int]]: List of n_atom_for_each_type.
    """
    n_atom_tuples = [(n, n_atom - n) for n in range(n_atom + 1)]
    return n_atom_tuples


def create_formula_dir_path(
    elements: Tuple[str, str], n_atom_for_each_element: Tuple[int, int]
) -> Path:
    """Create Path object of formula directory

    Args:
        elements (Tuple[str, str]): The elements included in system.
        n_atom_for_each_element (Tuple[int, int]): The number of atoms for each element.

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
