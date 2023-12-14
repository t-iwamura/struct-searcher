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
def cart_coords() -> NDArray:
    coords = np.array(
        [
            [0.805905937090300, 1.678409862356494, 0.060898389804236],
            [0.766172230955806, 1.283098321743793, 3.876241916957220],
            [1.144017839857946, 1.155195672150364, 2.839211381323941],
            [1.343643192752089, 1.321884927994224, 2.798906279231304],
            [1.716630828585983, 1.390143736128199, 3.019554356583979],
            [0.426452689433396, 1.855834387883555, 3.977128461006855],
            [1.531042801845348, 1.635065320329115, 2.349765536883962],
            [1.672440973408801, 1.227206941708905, 2.100560655593748],
            [2.108899027619434, 1.782553932931701, 4.013271337191338],
            [1.677670186487408, 0.553112971496572, 4.224160318206247],
            [1.005556520663251, 0.851272174408129, 0.004318707120197],
        ]
    )
    return coords


@pytest.fixture()
def potential_file() -> str:
    potential_file_path = (
        TESTS_DIR_PATH / "data" / "potentials" / "Ti-Al" / "gtinv-411" / "mlp.lammps"
    )
    return str(potential_file_path)
