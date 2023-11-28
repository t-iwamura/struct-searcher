from pathlib import Path

import click
from joblib import Parallel, delayed

from struct_searcher.bin import generate_input_files_for_relaxation, run_lammps


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("n_atom_for_each_type", type=int, nargs=-1)
@click.option("-p", "--potential_file", required=True, help="Path to mlp.lammps.")
def generate(n_atom_for_each_type, potential_file) -> None:
    generate_input_files_for_relaxation(n_atom_for_each_type, potential_file)


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
