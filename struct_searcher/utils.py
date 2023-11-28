from typing import List, Tuple


def create_n_atom_tuples(n_atom: int) -> List[Tuple[int, int]]:
    """Create n_atom tuples

    Args:
        n_atom (int): The number of atoms in unitcell.

    Returns:
        List[Tuple[int, int]]: List of n_atom_for_each_type.
    """
    n_atom_tuples = [(n, n_atom - n) for n in range(n_atom + 1)]
    return n_atom_tuples
