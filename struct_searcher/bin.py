from pathlib import Path

from lammps import lammps


def run_lammps(structure_dir_path: Path) -> None:
    """Run LAMMPS

    Args:
        structure_dir_path (Path): Path object of structure directory.
    """
    lmp = lammps()
    command_file_path = structure_dir_path / "in.lammps"
    lmp.file(str(command_file_path))
