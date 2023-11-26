from pathlib import Path
from typing import Optional

from numpy.typing import NDArray


def create_lammps_struct_file(
    xhi: float,
    yhi: float,
    zhi: float,
    xy: float,
    xz: float,
    yz: float,
    frac_coords: NDArray,
) -> str:
    """Create structure file for LAMMPS

    Args:
        xhi (float): The parameter about system.
        yhi (float): The parameter about system.
        zhi (float): The parameter about system.
        xy (float): The tilt parameter.
        xz (float): The tilt parameter.
        yz (float): The tilt parameter.
        frac_coords (NDArray): The fractional coordinates of all the atoms.
            The shape is (n_atom, 3).

    Returns:
        str: The content of a structure file.
    """
    # Create system section
    n_atom = frac_coords.shape[0]
    lines = [
        "# generated by struct-searcher",
        "",
        f"{n_atom} atoms",
        "1 atom types",
        "",
        f"0.0 {xhi:.15f} xlo xhi",
        f"0.0 {yhi:.15f} ylo yhi",
        f"0.0 {zhi:.15f} zlo zhi",
        "",
        f"{xy:.15f} {xz:.15f} {yz:.15f} xy xz yz",
        "",
    ]

    # Create Atoms section
    atoms_section = ["Atoms", ""]
    for i, coords in enumerate(frac_coords.tolist(), 1):
        coords_str = " ".join("{:.15f}".format(coord) for coord in coords)
        line = f"{i} 1 {coords_str}"
        atoms_section.append(line)
    atoms_section.append("")

    lines.extend(atoms_section)
    content = "\n".join(lines)

    return content


def create_lammps_command_file(
    potential_file: str, cwd_path: Optional[Path] = None
) -> str:
    """Create lammps command file

    Args:
        potential_file (str): The path of mlp.lammps.
        cwd_path (Optional[Path]): Path object of current working directory.
            Defaults to None.

    Returns:
        str: The content of lammps command file.
    """
    if cwd_path is None:
        cwd_path = Path.cwd()

    # Convert relative path to absolute path
    potential_file = str(Path(potential_file).resolve())
    initial_struct_file = str(cwd_path.resolve() / "initial_structure")
    final_struct_file = str(cwd_path.resolve() / "final_structure")

    # Read elements from potential
    with open(potential_file) as f:
        first_line = f.readline()

    elements = []
    for item in first_line.split(" "):
        if item == "#":
            break

        elements.append(item)
    elements_str = " ".join(elements)

    # Settings about relaxation
    etol = 0.0
    ftol = 1e-8
    maxiter = 1000
    maxeval = 100000
    pressure = 0.0

    lines = [
        "box tilt large",
        "atom_style atomic",
        "",
        "boundary p p p",
        f"read_data {initial_struct_file}",
        "",
        "pair_style polymlp",
        f"pair_coeff * * {potential_file} {elements_str}",
        "",
        "# What to monitor during minimization",
        "thermo 1",
        "thermo_style custom step temp pe etotal press fnorm",
        "thermo_modify norm no",
        "",
        "# Rebuild neighbor list at every timestep",
        "neigh_modify delay 0 every 1 check yes",
        "",
        "# Move atoms only",
        f"minimize {etol} {ftol} {maxiter} {maxeval}",
        "reset_timestep 0",
        "",
        "# Do isotropic volume relaxation",
        f"fix fiso all box/relax iso {pressure}",
        f"minimize {etol} {ftol} {maxiter} {maxeval}",
        "unfix fiso",
        "reset_timestep 0",
        "",
        "# Do anisotropic volume relaxation without shear",
        f"fix faniso all box/relax aniso {pressure}",
        f"minimize {etol} {ftol} {maxiter} {maxeval}",
        "unfix faniso",
        "reset_timestep 0",
        "",
        "# Do anisotropic volume relaxation with shear",
        f"fix ftri all box/relax tri {pressure}",
        f"minimize {etol} {ftol} {maxiter} {maxeval}",
        "unfix ftri",
        "reset_timestep 0",
        "",
        "# Output final structure",
        f"write_data {final_struct_file}",
        "",
    ]
    content = "\n".join(lines)

    return content
