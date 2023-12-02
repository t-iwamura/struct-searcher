import numpy as np
import pytest

from struct_searcher.struct import (
    convert_lattice_constants_to_niggli_cell,
    convert_niggli_cell_to_lattice_constants,
)


@pytest.mark.parametrize(
    ("niggli", "expected"),
    [
        ([4.0, 4.0, 4.0, 0.0, 0.0, 0.0], (2.0, 2.0, 2.0, 90.0, 90.0, 90.0)),
        ([25.0, 36.0, 81.0, 0.0, 0.0, 0.0], (5.0, 6.0, 9.0, 90.0, 90.0, 90.0)),
        ([4.0, 81.0, 1.0, 0.0, 0.0, -4.5], (2.0, 9.0, 1.0, 120.0, 90.0, 90.0)),
        ([49.0, 49.0, 4.0, 0.0, -7.0, 0.0], (7.0, 7.0, 2.0, 90.0, 120.0, 90.0)),
        ([9.0, 64.0, 25.0, -12.0, 0.0, 0.0], (3.0, 8.0, 5.0, 90.0, 90.0, 120.0)),
        (
            [
                64.0,
                9.0,
                16.0,
                0.8375879208600259,
                -0.5584770059930713,
                0.8370776849295027,
            ],
            (8.0, 3.0, 4.0, 86.0, 91.0, 88.0),
        ),
    ],
)
def test_convert_niggli_cell_to_lattice_constants(niggli, expected):
    np.testing.assert_allclose(
        convert_niggli_cell_to_lattice_constants(niggli),
        expected,
        rtol=0.0,
        atol=1e-09,
    )


@pytest.mark.parametrize(
    ("lattice_constants", "expected"),
    [
        ((2.0, 2.0, 2.0, 90.0, 90.0, 90.0), (4.0, 4.0, 4.0, 0.0, 0.0, 0.0)),
        ((5.0, 6.0, 9.0, 90.0, 90.0, 90.0), [25.0, 36.0, 81.0, 0.0, 0.0, 0.0]),
        ((2.0, 9.0, 1.0, 120.0, 90.0, 90.0), [4.0, 81.0, 1.0, 0.0, 0.0, -4.5]),
        ((7.0, 7.0, 2.0, 90.0, 120.0, 90.0), [49.0, 49.0, 4.0, 0.0, -7.0, 0.0]),
        ((3.0, 8.0, 5.0, 90.0, 90.0, 120.0), [9.0, 64.0, 25.0, -12.0, 0.0, 0.0]),
        (
            (8.0, 3.0, 4.0, 86.0, 91.0, 88.0),
            [
                64.0,
                9.0,
                16.0,
                0.8375879208600259,
                -0.5584770059930713,
                0.8370776849295027,
            ],
        ),
    ],
)
def test_convert_lattice_constants_to_niggli_cell(lattice_constants, expected):
    a, b, c, alpha, beta, gamma = lattice_constants
    np.testing.assert_allclose(
        convert_lattice_constants_to_niggli_cell(a, b, c, alpha, beta, gamma),
        expected,
        rtol=0.0,
        atol=1e-09,
    )
