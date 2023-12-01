from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from numpy.typing import NDArray

TESTS_DIR_PATH = Path(__file__).resolve().parent


@pytest.fixture()
def system_params() -> Dict[str, float]:
    params = {
        "xhi": 1.9632841787984097,
        "yhi": 1.7484482123237162,
        "zhi": 4.871325478627874,
        "xy": 0.005747308065752299,
        "xz": 0.41434324760446584,
        "yz": 0.262024857720561,
    }
    return params


@pytest.fixture()
def frac_coords() -> NDArray:
    coords = np.array(
        [
            [0.4050456661745165388, 0.9580690883003422087, 0.0125014003008867558],
            [0.2205162677133074567, 0.6146011287573603932, 0.7957263241726678649],
            [0.4580213682068485781, 0.5733522243207743729, 0.5828416503435269735],
            [0.5611641333797449116, 0.6699277106020067318, 0.5745677006209166660],
            [0.7414918128615224369, 0.7021793478251642240, 0.6198629859227778871],
            [0.0421592523686147302, 0.9390656753127170076, 0.8164366102112948775],
            [0.6755100744937648782, 0.8628641245474358845, 0.4823667700286431437],
            [0.7589883338864750195, 0.6372618810306311943, 0.4312092601509809331],
            [0.8976744141585206727, 0.8960420661578744062, 0.8238561259761628675],
            [0.6709685786735937185, 0.1863930705076654037, 0.8671480353220175630],
            [0.5105688454444539115, 0.4867401095835778291, 0.0008865568804927726],
        ]
    )
    return coords


@pytest.fixture()
def potential_file() -> str:
    potential_file_path = (
        TESTS_DIR_PATH / "data" / "potentials" / "Al-Cu" / "gtinv-257" / "mlp.lammps"
    )
    return str(potential_file_path)


@pytest.fixture()
def dumped_lammps_command_content() -> str:
    lammps_command_file_path = TESTS_DIR_PATH / "data" / "commands" / "in.lammps"
    with lammps_command_file_path.open("r") as f:
        content = f.read()
    return content
