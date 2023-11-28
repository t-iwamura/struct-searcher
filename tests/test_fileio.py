from struct_searcher.fileio import create_lammps_command_file, create_lammps_struct_file


def test_create_lammps_struct_file(
    system_params, frac_coords, dumped_lammps_struct_content
):
    content = create_lammps_struct_file(
        system_params["xhi"],
        system_params["yhi"],
        system_params["zhi"],
        system_params["xy"],
        system_params["xz"],
        system_params["yz"],
        frac_coords,
        [7, 4],
    )
    assert content == dumped_lammps_struct_content


def test_create_lammps_command_file(
    potential_file, struct_dir_path, dumped_lammps_command_content
):
    content = create_lammps_command_file(
        potential_file, output_dir_path=struct_dir_path
    )
    assert content == dumped_lammps_command_content
