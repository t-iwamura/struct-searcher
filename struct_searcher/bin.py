from pathlib import Path
from typing import List

from lammps import lammps
from pymatgen.io.lammps.data import LammpsData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from struct_searcher.fileio import (
    create_job_script,
    create_lammps_command_file,
    create_lammps_struct_file_from_structure,
    create_sample_struct_file,
    parse_lammps_log,
)
from struct_searcher.utils import calc_begin_id_of_dir, create_formula_dir_path


def generate_input_files_for_relaxation(
    elements: List[str],
    n_atom_for_each_element: List[int],
    potential_file: str,
    structure_id: str,
    g_max: float,
) -> None:
    """Generate input files for relaxation

    Args:
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        potential_file (str): Path to a potential file.
        structure_id (str): The ID of a sample structure.
        g_max (float): The parameter to control volume maximum.
    """
    # Make output directory
    formula_dir_path = create_formula_dir_path(elements, n_atom_for_each_element)
    output_dir_path = formula_dir_path / "multi_start" / structure_id
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
) -> None:
    """Write job script

    Args:
        elements (List[str]): List of element included in system.
        n_atom_for_each_element (List[int]): The number of atoms for each element.
        begin_sid (int): The begin ID of a structure.
        relax_once (bool, optional): Whether to relax just once or not.
            Defaults to False.
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
        job_name=formula_dir_path.name, first_sid=begin_sid, relax_once=relax_once
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


def run_lammps(structure_dir_path: Path, relaxation_id: str = "00") -> None:
    """Run LAMMPS

    Args:
        structure_dir_path (Path): Path object of structure directory.
        relaxation_id (str, optional): The ID of relaxation. Defaults to '00'.
    """
    # Settings about log
    log_file_path = structure_dir_path / "log.lammps"
    if relaxation_id != "00":
        log_file_path = structure_dir_path / f"log_{relaxation_id}.lammps"
    lmp = lammps(cmdargs=["-log", str(log_file_path), "-screen", "none"])

    command_file_path = structure_dir_path / "in.lammps"
    if relaxation_id != "00":
        command_file_path = structure_dir_path / f"in_{relaxation_id}.lammps"
    lmp.file(str(command_file_path))


def relax_step_by_step(structure_dir_path: Path) -> None:
    """Relax a structure step by step

    Args:
        structure_dir_path (Path): Object of structure directory.
    """
    try:
        # Do easy relaxation
        run_lammps(structure_dir_path, relaxation_id="01")

        calc_stats = parse_lammps_log(str(structure_dir_path / "log_01.lammps"))
        if calc_stats["criterion"] != "force tolerance":
            return

        # Refine the structure after 1st relaxation
        structure = LammpsData.from_file(
            str(structure_dir_path / "final_structure_01"), atom_style="atomic"
        ).structure
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-05, angle_tolerance=-1.0)
        refined_structure = analyzer.get_refined_structure()

        content = create_lammps_struct_file_from_structure(refined_structure)
        struct_file_path = structure_dir_path / "initial_structure_02"
        with struct_file_path.open("w") as f:
            f.write(content)

        # Do hard relaxation
        run_lammps(structure_dir_path, relaxation_id="02")
    except Exception as e:
        err_log_path = structure_dir_path / "err.log"
        with err_log_path.open("w") as f:
            print(e, file=f)
