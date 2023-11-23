import click

from struct_searcher.struct import create_sample_struct_file


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("g_max", type=float)
def generate(g_max) -> None:
    n_atom = 4
    lammps_struct_file_content = create_sample_struct_file(g_max, n_atom)
    with open("initial_structure", "w") as f:
        f.write(lammps_struct_file_content)
