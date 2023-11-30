import json
import re
from math import pi
from pathlib import Path

import click
from joblib import Parallel, delayed

from struct_searcher.bin import generate_input_files_for_relaxation, run_lammps
from struct_searcher.data import ATOM_INFO
from struct_searcher.fileio import read_elements
from struct_searcher.utils import create_formula_dir_path, create_n_atom_tuples

INPUTS_DIR_PATH = Path.home() / "struct-searcher" / "data" / "inputs"
PROCESSING_DIR_PATH = Path.home() / "struct-searcher" / "data" / "processing"


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("system_name")
@click.option(
    "-n", "--n_atom", type=int, required=True, help="The number of atoms in unitcell."
)
def generate(system_name, n_atom) -> None:
    """Generate 1000 sample structures for all the compositions"""
    # Check a recommended potential for system
    potential_id_json_path = PROCESSING_DIR_PATH / "potential_id.json"
    with potential_id_json_path.open("r") as f:
        potential_ids = json.load(f)
    potential_file_path = (
        INPUTS_DIR_PATH
        / "potentials"
        / system_name
        / potential_ids[system_name]
        / "mlp.lammps"
    )

    # Calculate g_max
    elements = read_elements(system_name)
    d = max(ATOM_INFO[e]["distance"] for e in elements)
    g_max = (n_atom * 10 * 4 * pi * (0.5 * d) ** 3 / 3) ** (2 / 3)

    n_atom_tuples = create_n_atom_tuples(n_atom)
    n_structure = 1000
    for n_atom_for_each_element in n_atom_tuples:
        # Calculate the begin ID of a sample structure
        formula_dir_path = create_formula_dir_path(elements, n_atom_for_each_element)
        existing_sids = sorted(
            int(p.name)
            for p in formula_dir_path.glob("*")
            if re.search(r".*/\d{5}", str(p))
        )
        if len(existing_sids) == 0:
            sid_begin = 1
        else:
            sid_begin = existing_sids[-1] + 1

        _ = Parallel(n_jobs=-1, verbose=1)(
            delayed(generate_input_files_for_relaxation)(
                elements,
                n_atom_for_each_element,
                str(potential_file_path),
                str(sid_begin + i).zfill(5),
                g_max,
            )
            for i in range(n_structure)
        )


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
