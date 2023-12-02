import json
from pathlib import Path
from typing import List, Optional

from numpy.typing import NDArray

from struct_searcher.data import load_atom_info

POTENTIALS_DIR_PATH = Path.home() / "struct-searcher" / "data" / "inputs" / "potentials"


def read_elements(
    system_name: str, potentials_dir_path: Optional[Path] = None
) -> List[str]:
    """Read elements in the order defined in mlp.lammps

    Args:
        system_name (str): The name of a system.
        potentials_dir_path (Optional[Path], optional):
            Path object of potentials directory. Defaults to None.

    Returns:
        List[str]: The elements keeping the order.
    """
    if potentials_dir_path is None:
        potentials_dir_path = POTENTIALS_DIR_PATH

    # Read IDs of recommended potentials
    potential_id_json_path = potentials_dir_path / "potential_id.json"
    with potential_id_json_path.open("r") as f:
        potential_ids = json.load(f)

    # Read elements from mlp.lammps
    potential_file_path = (
        potentials_dir_path / system_name / potential_ids[system_name] / "mlp.lammps"
    )
    with potential_file_path.open("r") as f:
        first_line = f.readline()

    elements = []
    for item in first_line.split():
        if item == "#":
            break

        elements.append(item)

    return elements


def create_lammps_struct_file(
    xhi: float,
    yhi: float,
    zhi: float,
    xy: float,
    xz: float,
    yz: float,
    frac_coords: NDArray,
    elements: List[str],
    n_atom_for_each_element: List[int],
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
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.

    Returns:
        str: The content of a structure file.
    """
    # Remove elements which don't exist
    elements = [
        e for e, n_atom in zip(elements, n_atom_for_each_element) if n_atom != 0
    ]
    n_atom_for_each_element = [n for n in n_atom_for_each_element if n != 0]

    # Create system section
    n_atom = frac_coords.shape[0]
    n_type = len(n_atom_for_each_element)
    lines = [
        "# generated by struct-searcher",
        "",
        f"{n_atom} atoms",
        f"{n_type} atom types",
        "",
        f"0.0 {xhi:.15f} xlo xhi",
        f"0.0 {yhi:.15f} ylo yhi",
        f"0.0 {zhi:.15f} zlo zhi",
        "",
        f"{xy:.15f} {xz:.15f} {yz:.15f} xy xz yz",
        "",
    ]

    # Create Masses section
    masses_section = ["Masses", ""]
    atom_info = load_atom_info()
    for i, e in enumerate(elements, 1):
        masses_section.append(f"{i} {atom_info[e]['mass']:.8f}")
    masses_section.append("")

    # Create type list for all the atoms
    types = [
        i for i, n_atom in enumerate(n_atom_for_each_element, 1) for _ in range(n_atom)
    ]

    # Create Atoms section
    atoms_section = ["Atoms", ""]
    for i, (atom_type, coords) in enumerate(zip(types, frac_coords.tolist()), 1):
        coords_str = " ".join("{:.15f}".format(coord) for coord in coords)
        line = f"{i} {atom_type} {coords_str}"
        atoms_section.append(line)
    atoms_section.append("")

    lines.extend(masses_section)
    lines.extend(atoms_section)
    content = "\n".join(lines)

    return content


def create_lammps_command_file(
    potential_file: str,
    elements: List[str],
    n_atom_for_each_element: List[int],
    output_dir_path: Path,
    ftol: float,
    relaxation_id: str = "01",
) -> str:
    """Create lammps command file

    Args:
        potential_file (str): The path of mlp.lammps.
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        output_dir_path (Path): Path object of output directory.
        ftol (float): The tolerance for global force vector.
        relaxation_id (str, optional): The ID of relaxation. Defaults to '01'.

    Returns:
        str: The content of lammps command file.
    """
    # Convert relative path to absolute path
    potential_file = str(Path(potential_file).resolve())
    initial_struct_file = str(
        output_dir_path.resolve() / f"initial_structure_{relaxation_id}"
    )
    final_struct_file = str(
        output_dir_path.resolve() / f"final_structure_{relaxation_id}"
    )

    # Choose the element which exists
    elements_str = " ".join(
        e for e, n in zip(elements, n_atom_for_each_element) if n != 0
    )

    # Settings about relaxation
    etol = 0.0
    maxiter = 50000
    maxeval = 500000
    pressure = 0.0

    lines = [
        "units metal",
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
        "neigh_modify delay 0 every 1 check yes one 100000 page 1000000",
        "",
    ]

    n_atom = sum(n_atom_for_each_element)
    if n_atom < 10:
        relaxation_section = [
            "# Do relaxation with a bunch of degrees of freedom",
            f"fix ftri all box/relax tri {pressure}",
            f"minimize 0.0 {ftol} {maxiter} {maxeval}",
            "",
        ]
    else:
        relaxation_section = [
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
        ]

    save_section = [
        "# Output final structure",
        f"write_data {final_struct_file}",
        "",
    ]

    lines.extend(relaxation_section)
    lines.extend(save_section)
    content = "\n".join(lines)

    return content


def create_job_script(job_name: str, first_sid: int) -> str:
    """Create job script

    Args:
        job_name (str): The name of a job.
        first_sid (int): The ID of first structure.

    Returns:
        str: The content of a job script.
    """
    # Create pattern of sample structure directories
    n_structure = 1000
    last_sid = first_sid + n_structure - 1
    dir_pattern = "".join(
        ["{", str(first_sid).zfill(5), "..", str(last_sid).zfill(5), "}"]
    )

    lines = [
        "#!/bin/zsh",
        f"#SBATCH -J {job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH -e err.log",
        "#SBATCH -o std.log",
        "#SBATCH --open-mode=append",
        "",
        ". ~/.zprofile",
        ". ~/.zshrc",
        "pyenv activate structural_search",
        f"struct-searcher relax-by-mlp {dir_pattern}",
        "",
    ]
    content = "\n".join(lines)

    return content
