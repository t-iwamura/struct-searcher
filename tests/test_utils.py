from pathlib import Path

import pytest

from struct_searcher.utils import calc_begin_id_of_dir, create_n_atom_lists

TESTS_DIR_PATH = Path(__file__).resolve().parent
JOB_SCRIPTS_DIR_PATH = TESTS_DIR_PATH / "data" / "job_scripts"


@pytest.mark.parametrize(
    ("n_atom", "expected"),
    [
        (1, [[0, 1], [1, 0]]),
        (2, [[0, 2], [1, 1], [2, 0]]),
        (3, [[0, 3], [1, 2], [2, 1], [3, 0]]),
    ],
)
def test_create_n_atom_lists(n_atom, expected):
    assert create_n_atom_lists(n_atom) == expected


def test_calc_begin_id_of_dir():
    assert calc_begin_id_of_dir(JOB_SCRIPTS_DIR_PATH, 3) == 2
