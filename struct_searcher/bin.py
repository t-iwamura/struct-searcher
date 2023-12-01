from pathlib import Path
from typing import List

from lammps import lammps

from struct_searcher.fileio import create_lammps_command_file
from struct_searcher.struct import create_sample_struct_file
from struct_searcher.utils import create_formula_dir_path


def generate_input_files_for_relaxation(
    elements: List[str],
    n_atom_for_each_element: List[int],
    potential_file: str,
    structure_id: str,
    g_max: float,
) -> None:
    """Generate input files for relaxation

    Args:
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        potential_file (str): Path to a potential file.
        structure_id (str): The ID of a sample structure.
        g_max (float): The parameter to control volume maximum.
    """
    # Make output directory
    formula_dir_path = create_formula_dir_path(elements, n_atom_for_each_element)
    output_dir_path = formula_dir_path / "multi_start" / structure_id
    output_dir_path.mkdir(parents=True)

    # Write sample structure file
    content = create_sample_struct_file(g_max, elements, n_atom_for_each_element)
    struct_file_path = output_dir_path / "initial_structure"
    with struct_file_path.open("w") as f:
        f.write(content)

    # Write lammps command file
    content = create_lammps_command_file(
        potential_file, elements, n_atom_for_each_element, output_dir_path
    )
    command_file_path = output_dir_path / "in.lammps"
    with command_file_path.open("w") as f:
        f.write(content)


def run_lammps(structure_dir_path: Path) -> None:
    """Run LAMMPS

    Args:
        structure_dir_path (Path): Path object of structure directory.
    """
    # Settings about log
    log_file_path = structure_dir_path / "log.lammps"
    lmp = lammps(cmdargs=["-log", str(log_file_path), "-screen", "none"])

    command_file_path = structure_dir_path / "in.lammps"
    lmp.file(str(command_file_path))
