import click

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
