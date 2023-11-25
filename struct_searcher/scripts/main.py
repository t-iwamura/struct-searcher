import click

from struct_searcher.fileio import create_lammps_command_file
from struct_searcher.struct import create_sample_struct_file


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("n_atom_for_each_type", type=int, nargs=-1)
@click.option("--g_max", default=30.0, help="The parameter, g_max.")
@click.option("-p", "--potential_file", required=True, help="Path to mlp.lammps.")
def generate(n_atom_for_each_type, g_max, potential_file) -> None:
    lammps_struct_file_content = create_sample_struct_file(g_max, n_atom_for_each_type)
    with open("initial_structure", "w") as f:
        f.write(lammps_struct_file_content)

    lammps_command_file_content = create_lammps_command_file(potential_file)
    with open("in.lammps", "w") as f:
        f.write(lammps_command_file_content)
