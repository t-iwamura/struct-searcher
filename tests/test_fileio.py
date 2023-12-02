from pathlib import Path

import pytest

from struct_searcher.fileio import (
    create_lammps_command_file,
    create_lammps_struct_file,
    parse_lammps_log,
    read_elements,
)

TESTS_DIR_PATH = Path(__file__).resolve().parent
STRUCT_DIR_PATH = TESTS_DIR_PATH / "data" / "structures"
POTENTIALS_DIR_PATH = TESTS_DIR_PATH / "data" / "potentials"
COMMANDS_DIR_PATH = TESTS_DIR_PATH / "data" / "commands"
LOGS_DIR_PATH = TESTS_DIR_PATH / "data" / "logs"


@pytest.mark.parametrize(
    ("system_name", "expected"), [("Al-Cu", ["Al", "Cu"]), ("Na-Sn", ["Na", "Sn"])]
)
def test_read_elements(system_name, expected):
    assert read_elements(system_name, POTENTIALS_DIR_PATH) == expected


@pytest.fixture()
def dumped_lammps_struct_content(request):
    lammps_struct_file_path = STRUCT_DIR_PATH / request.param / "lammps_structure"
    with lammps_struct_file_path.open("r") as f:
        content = f.read()
    return content


@pytest.mark.parametrize(
    ("n_atom_for_each_elements", "dumped_lammps_struct_content"),
    [([7, 4], "Ti7-Al4"), ([11, 0], "Ti"), ([0, 11], "Al")],
    indirect=["dumped_lammps_struct_content"],
)
def test_create_lammps_struct_file(
    system_params, frac_coords, n_atom_for_each_elements, dumped_lammps_struct_content
):
    content = create_lammps_struct_file(
        system_params["xhi"],
        system_params["yhi"],
        system_params["zhi"],
        system_params["xy"],
        system_params["xz"],
        system_params["yz"],
        frac_coords,
        ["Ti", "Al"],
        n_atom_for_each_elements,
    )
    assert content == dumped_lammps_struct_content


@pytest.fixture()
def dumped_lammps_command_content(request):
    lammps_command_file_path = COMMANDS_DIR_PATH / request.param / "in.lammps"
    with lammps_command_file_path.open("r") as f:
        content = f.read()
    return content


@pytest.mark.parametrize(
    ("n_atom_for_each_element", "dumped_lammps_command_content"),
    [
        ([7, 4], "Ti7-Al4"),
        ([11, 0], "Ti"),
        ([0, 11], "Al"),
        ([4, 6], "Ti4-Al6"),
        ([2, 7], "Ti2-Al7"),
    ],
    indirect=["dumped_lammps_command_content"],
)
def test_create_lammps_command_file(
    potential_file, n_atom_for_each_element, dumped_lammps_command_content
):
    content = create_lammps_command_file(
        potential_file,
        ["Ti", "Al"],
        n_atom_for_each_element,
        output_dir_path=STRUCT_DIR_PATH,
        ftol=1e-08,
    )
    assert content == dumped_lammps_command_content


def test_parse_lammps_log(calc_stats):
    assert parse_lammps_log(str(LOGS_DIR_PATH / "log.lammps")) == calc_stats
