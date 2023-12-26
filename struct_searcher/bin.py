import json
import shutil
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from lammps import lammps
from numpy.typing import NDArray
from pymatgen.io.lammps.data import LammpsData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from struct_searcher.fileio import (
    create_job_script,
    create_lammps_command_file,
    create_lammps_struct_file,
    create_lammps_struct_file_from_structure,
    create_sample_struct_file,
    create_static_lammps_command_file,
)
from struct_searcher.struct import find_same_structure
from struct_searcher.utils import (
    calc_begin_id_of_dir,
    check_previous_relaxation,
    create_formula_dir_path,
)


def generate_input_files_for_relaxation(
    elements: List[str],
    n_atom_for_each_element: List[int],
    potential_file: str,
    structure_id: str,
    g_max: float,
    output_dir_id: str,
) -> None:
    """Generate input files for relaxation

    Args:
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        potential_file (str): Path to a potential file.
        structure_id (str): The ID of a sample structure.
        g_max (float): The parameter to control volume maximum.
        output_dir_id (str): The ID of output directory.
    """
    # Make output directory
    formula_dir_path = create_formula_dir_path(elements, n_atom_for_each_element)
    output_dir_path = formula_dir_path / "multi_start" / structure_id / output_dir_id
    output_dir_path.mkdir(parents=True)

    # Write sample structure file
    content = create_sample_struct_file(g_max, elements, n_atom_for_each_element)
    struct_file_path = output_dir_path / "initial_structure_01"
    with struct_file_path.open("w") as f:
        f.write(content)

    # Write lammps command files
    content = create_lammps_command_file(
        potential_file,
        elements,
        n_atom_for_each_element,
        output_dir_path,
        ftol=1e-03,
        relaxation_id="01",
    )
    command_file_path = output_dir_path / "in_01.lammps"
    with command_file_path.open("w") as f:
        f.write(content)

    content = create_lammps_command_file(
        potential_file,
        elements,
        n_atom_for_each_element,
        output_dir_path,
        ftol=1e-08,
        relaxation_id="02",
    )
    command_file_path = output_dir_path / "in_02.lammps"
    with command_file_path.open("w") as f:
        f.write(content)


def write_job_script(
    elements: List[str],
    n_atom_for_each_element: List[int],
    begin_sid: int,
    relax_once: bool = False,
    output_dir_id: str = "01",
) -> None:
    """Write job script

    Args:
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        begin_sid (int): The begin ID of a structure.
        relax_once (bool, optional): Whether to relax just once or not.
            Defaults to False.
        output_dir_id (str, optional): The ID of output directory. Defaults to '01'.
    """
    # Make job_scripts directory
    formula_dir_path = create_formula_dir_path(elements, n_atom_for_each_element)
    job_scripts_dir_path = formula_dir_path / "job_scripts"
    if not job_scripts_dir_path.exists():
        job_scripts_dir_path.mkdir()

    # Make output directory
    begin_jid = calc_begin_id_of_dir(job_scripts_dir_path, n_digit=3)
    output_dir_path = job_scripts_dir_path / str(begin_jid).zfill(3)
    output_dir_path.mkdir()

    content = create_job_script(
        job_name=formula_dir_path.name,
        first_sid=begin_sid,
        relax_once=relax_once,
        output_dir_id=output_dir_id,
    )
    job_script_path = output_dir_path / "job.sh"
    with job_script_path.open("w") as f:
        f.write(content)


def generate_new_lammps_command_file(
    structure_dir_path: Path,
    ftol: float,
    elements: List[str],
    n_atom_for_each_element: List[int],
    potential_file: str,
) -> None:
    """Generate new lammps command file

    Args:
        structure_dir_path (Path): Object of structure directory.
        ftol (float): The tolerance for global force vector.
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        potential_file (str): Path to a potential file.
    """
    # Make output directory
    begin_oid = calc_begin_id_of_dir(structure_dir_path, n_digit=2)
    output_dir_path = structure_dir_path / str(begin_oid).zfill(2)
    output_dir_path.mkdir()

    # Write lammps command file
    content = create_lammps_command_file(
        potential_file,
        elements,
        n_atom_for_each_element,
        output_dir_path,
        ftol,
        relaxation_id="02",
        input_dir_path=structure_dir_path / "01",
    )
    command_file_path = output_dir_path / "in_02.lammps"
    with command_file_path.open("w") as f:
        f.write(content)


def run_lammps(
    command_file: str,
    log_file: str = "log.lammps",
    save_log: bool = True,
) -> Dict[str, float]:
    """Run LAMMPS

    Args:
        command_file (str): Path to a LAMMPS command file.
        log_file (str, optional): Path to a LAMMPS log file. Defaults to "log.lammps".
        save_log (bool, optional): Whether to save log or not. Defaults to True.

    Returns:
        Dict[str, float]: Dict about the calculation result.
    """
    calc_stats = {"energy": 1e10}
    try:
        if save_log:
            lmp = lammps(cmdargs=["-log", log_file, "-screen", "none"])
        else:
            lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp.file(command_file)

        # Extract quantity
        n_atom = lmp.get_natoms()
        lmp.command("variable energy equal pe")
        lmp.command("run 0")
        calc_stats["energy"] = lmp.extract_variable("energy", None, 0)
        calc_stats["energy_per_atom"] = calc_stats["energy"] / n_atom
    except Exception:
        err_log_path = Path(log_file).parent / "err.log"
        with err_log_path.open("a") as f:
            print(traceback.format_exc(), file=f)

    return calc_stats


def run_lammps_as_one_cycle(
    output_dir_path: Path,
    relaxation_id: str,
    cycle_id: str,
) -> str:
    """Run LAMMPS as one cycle

    Args:
        output_dir_path (Path): Object of output directory.
        relaxation_id (str): The ID of relaxation.
        cycle_id (str): The ID of relaxation cycle.

    Returns:
        str: The result status of LAMMPS calculation.
    """
    command_file_path = output_dir_path / f"in_{relaxation_id}.lammps"
    log_file_path = output_dir_path / f"log_{relaxation_id}.lammps"
    calc_stats = run_lammps(str(command_file_path), str(log_file_path))

    # Copy log file and dumped structure file
    old_log_file_path = output_dir_path / f"log_{relaxation_id}.lammps"
    new_log_file_path = output_dir_path / f"log_{relaxation_id}-{cycle_id}.lammps"
    shutil.copy(old_log_file_path, new_log_file_path)

    old_final_structure_path = output_dir_path / f"final_structure_{relaxation_id}"
    if old_final_structure_path.exists():
        new_final_structure_path = (
            output_dir_path / f"final_structure_{relaxation_id}-{cycle_id}"
        )
        shutil.copy(old_final_structure_path, new_final_structure_path)

    # Copy and prepare initial structure
    old_initial_structure_path = output_dir_path / f"initial_structure_{relaxation_id}"
    new_initial_structure_path = (
        output_dir_path / f"initial_structure_{relaxation_id}-{cycle_id}"
    )
    shutil.copy(old_initial_structure_path, new_initial_structure_path)
    if old_final_structure_path.exists():
        shutil.copy(old_final_structure_path, old_initial_structure_path)

    # Judge if relaxation should be continued or not
    try:
        result_status = check_previous_relaxation(
            calc_stats, output_dir_path, relaxation_id
        )
    except Exception:
        result_status = "STOP: an error is raised"
        err_log_path = output_dir_path / "err.log"
        with err_log_path.open("a") as f:
            print(traceback.format_exc(), file=f)

    # Save energy and result status
    json_path = output_dir_path / "calc_stats.json"
    if json_path.exists():
        with json_path.open("r") as f:
            calc_stats_saved = json.load(f)
    else:
        calc_stats_saved = {}

    if cycle_id == "01":
        calc_stats_saved[f"energy_{relaxation_id}_per_atom"] = []
        calc_stats_saved[f"result_status_{relaxation_id}"] = []

    calc_stats_saved[f"energy_{relaxation_id}_per_atom"].append(
        calc_stats["energy_per_atom"]
    )
    calc_stats_saved[f"result_status_{relaxation_id}"].append(result_status)

    with json_path.open("w") as f:
        json.dump(calc_stats_saved, f, indent=4)

    return result_status


def relax_step_by_step(structure_dir_path: Path, output_dir_id: str) -> None:
    """Relax a structure step by step

    Args:
        structure_dir_path (Path): Object of structure directory.
        output_dir_id (str): The ID of output directory.
    """
    max_iteration = 10
    output_dir_path = structure_dir_path / output_dir_id

    # Do easy relaxation
    for i in range(max_iteration):
        result_status = run_lammps_as_one_cycle(
            output_dir_path,
            relaxation_id="01",
            cycle_id=str(i + 1).zfill(2),
        )

        if "STOP" in result_status:
            return
        elif result_status != "UNFINISHED":
            break

    # Refine the structure after 1st relaxation
    try:
        structure = LammpsData.from_file(
            str(output_dir_path / "final_structure_01"), atom_style="atomic"
        ).structure
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-05, angle_tolerance=-1.0)
        refined_structure = analyzer.get_refined_structure()

        content = create_lammps_struct_file_from_structure(refined_structure)
        struct_file_path = output_dir_path / "initial_structure_02"
        with struct_file_path.open("w") as f:
            f.write(content)
    except Exception:
        err_log_path = output_dir_path / "err.log"
        with err_log_path.open("a") as f:
            print(traceback.format_exc(), file=f)
        return

    # Do hard relaxation
    for i in range(max_iteration):
        result_status = run_lammps_as_one_cycle(
            output_dir_path,
            relaxation_id="02",
            cycle_id=str(i + 1).zfill(2),
        )

        if result_status != "UNFINISHED":
            return


def _calc_diatom_energy(
    potential_file: str,
    d: float,
    elements: List[str],
    n_atom_for_each_element: List[int],
    dmax: float,
) -> float:
    """Internal implementation of calc_diatom_energy()

    Args:
        potential_file (str): Path to a potential file.
        d (float): The interatomic distance.
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        dmax (float): The maximum of interatomic distance.

    Returns:
        float: Total energy when keeping the distance between two atoms.
    """
    xhi = dmax + 1
    frac_coords = np.array([[0.1, 0.1, 0.1], [0.1 + d, 0.1, 0.1]])

    # Write structure file for LAMMPS
    struct_file_object = NamedTemporaryFile(mode="w")
    content = create_lammps_struct_file(
        xhi, xhi, xhi, frac_coords, elements, n_atom_for_each_element
    )
    struct_file_object.write(content)
    struct_file_object.seek(0)

    # Write LAMMPS command file
    command_file_object = NamedTemporaryFile(mode="w")
    content = create_static_lammps_command_file(
        potential_file,
        elements,
        n_atom_for_each_element,
        struct_file=struct_file_object.name,
    )
    command_file_object.write(content)
    command_file_object.seek(0)

    calc_stats = run_lammps(command_file_object.name, save_log=False)

    command_file_object.close()
    struct_file_object.close()

    return calc_stats["energy_per_atom"]


def calc_diatom_energy(
    potential_file: str, elements: List[str], n_atom_for_each_element: List[int]
) -> Tuple[NDArray, NDArray]:
    """Calculate diatom energy

    Args:
        potential_file (str): Path to a potential file.
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.

    Returns:
        Tuple[NDArray, NDArray]: NumPy array of interatomic distance and
            NumPy array of total energy.
    """
    dmax = 6.0
    dmin = 0.02
    dsteps = 300
    darray = np.linspace(dmin, dmax, dsteps)

    energies = np.array(
        Parallel(n_jobs=-1, verbose=1)(
            delayed(_calc_diatom_energy)(
                potential_file, d, elements, n_atom_for_each_element, dmax
            )
            for d in darray
        )
    )

    return darray, energies


def extract_unique_structures(
    structure_ids: List[str],
    root_dir_path: Path,
) -> Tuple[List[float], List[str], List[List[str]]]:
    """Extract unique structures from relaxed structures

    Args:
        structure_ids (List[str]): List of structure ID.
        root_dir_path (Path): Object of root directory.

    Returns:
        Tuple[List[float], List[str], List[List[str]]]: Energies of unique structures,
            space group symbol of unique structures and IDs of duplicate structures.
    """
    energies: List[float] = []
    space_groups: List[str] = []
    duplicate_structure_ids: List[List[str]] = []
    for structure_id in structure_ids:
        calc_dir_path = root_dir_path / structure_id / "01"
        struct_file_path = calc_dir_path / "final_structure_02"
        if not struct_file_path.exists():
            continue

        sid, energy, space_group = find_same_structure(
            calc_dir_path, energies, space_groups
        )
        if sid == -1:
            continue
        else:
            duplicate_structure_ids[sid].append(structure_id)

        energies.append(energy)
        space_groups.append(space_group)
        duplicate_structure_ids.append([structure_id])

    return energies, space_groups, duplicate_structure_ids


def analyze_duplicate_structures(
    structure_ids: List[str],
    root_dir_path: Path,
) -> None:
    """Analyze if structures are duplicate or not

    Args:
        structure_ids (List[str]): List of structure ID.
        root_dir_path (Path): Object of root directory.
    """
    energies, space_groups, duplicate_structure_ids = extract_unique_structures(
        structure_ids, root_dir_path
    )

    # Sort objects in terms of energy
    indices = [i for i in range(len(energies))]
    indices.sort(key=lambda i: energies[i])
    energies = [energies[i] for i in indices]
    space_groups = [space_groups[i] for i in indices]
    duplicate_structure_ids = [duplicate_structure_ids[i] for i in indices]

    dft_dir_path = root_dir_path.parent.resolve() / "dft"
    if not dft_dir_path.exists():
        dft_dir_path.mkdir()

    # Write result as output files
    n_structure = len(energies)
    for i in range(n_structure):
        # Make output directory
        begin_sid = calc_begin_id_of_dir(dft_dir_path, n_digit=5)
        output_dir_path = dft_dir_path / str(begin_sid + i).zfill(5)
        output_dir_path.mkdir()

        duplicate_structure_ids[i].append("")
        duplicate_structure_ids_txt_path = (
            output_dir_path / "duplicate_structure_ids.txt"
        )
        with duplicate_structure_ids_txt_path.open("w") as f:
            f.write("\n".join(duplicate_structure_ids[i]))

        struct_info = {"energy": energies[i], "space_group": space_groups[i]}
        struct_info_json_path = output_dir_path / "struct_info.json"
        with struct_info_json_path.open("w") as f:
            json.dump(struct_info, f, indent=4)

        # Refine a structure
        struct_file_path = (
            root_dir_path / duplicate_structure_ids[i][0] / "01" / "final_structure_02"
        )
        structure = LammpsData.from_file(
            str(struct_file_path), atom_style="atomic"
        ).structure
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-05, angle_tolerance=-1.0)
        structure = analyzer.get_refined_structure()

        poscar_path = output_dir_path / "POSCAR"
        structure.to(str(poscar_path), fmt="poscar")
