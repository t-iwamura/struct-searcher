from struct_searcher.fileio import create_lammps_struct_file


def test_create_lammps_struct_file(
    system_params, frac_coords, dumped_lammps_struct_content
):
    lammps_struct_content = create_lammps_struct_file(
        system_params["xhi"],
        system_params["yhi"],
        system_params["zhi"],
        system_params["xy"],
        system_params["xz"],
        system_params["yz"],
        frac_coords,
    )
    assert lammps_struct_content == dumped_lammps_struct_content
