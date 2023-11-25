from pathlib import Path

import click
from joblib import Parallel, delayed

from struct_searcher.bin import run_lammps
from struct_searcher.fileio import create_lammps_command_file
from struct_searcher.struct import create_sample_struct_file


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("g_max", type=float)
@click.option("-p", "--potential_file", required=True, help="Path to mlp.lammps")
def generate(g_max, potential_file) -> None:
    n_atom = 4
    lammps_struct_file_content = create_sample_struct_file(g_max, n_atom)
    with open("initial_structure", "w") as f:
        f.write(lammps_struct_file_content)

    lammps_command_file_content = create_lammps_command_file(potential_file)
    with open("in.lammps", "w") as f:
        f.write(lammps_command_file_content)


@main.command()
@click.argument("structure_ids", nargs=-1)
def relax_by_mlp(structure_ids) -> None:
    """Relax multiple structures by polymlp"""
    structural_search_dir_path = (
        Path.cwd().parent.parent.resolve() / "structural_search"
    )
    structure_dir_path_list = [
        structural_search_dir_path / structure_id for structure_id in structure_ids
    ]

    # Run relaxation of multiple structures by polymlp
    _ = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_lammps)(path) for path in structure_dir_path_list
    )
