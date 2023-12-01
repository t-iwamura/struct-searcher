import pytest

from struct_searcher.utils import create_n_atom_lists


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
