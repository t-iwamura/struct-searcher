from pathlib import Path
from typing import List

from lammps import lammps

from struct_searcher.fileio import create_lammps_command_file
from struct_searcher.struct import create_sample_struct_file


def generate_input_files_for_relaxation(
    n_atom_for_each_type: List[int], potential_file: str, g_max: float = 30.0
) -> None:
    """Generate input files for relaxation

    Args:
        n_atom_for_each_type (List[int]): The number of atoms for each type.
        potential_file (str): Path to a potential file.
        g_max (float, optional): The parameter, g_max. Defaults to 30.0.
    """
    # Write sample structure file
    content = create_sample_struct_file(g_max, n_atom_for_each_type)
    with open("initial_structure", "w") as f:
        f.write(content)

    # Write lammps command file
    content = create_lammps_command_file(potential_file)
    with open("in.lammps", "w") as f:
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
