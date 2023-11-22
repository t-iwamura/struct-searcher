import random
from typing import List


def create_niggli_cell(g_max: float) -> List[float]:
    """Create Niggli reduced cell

    Args:
        g_max (float): The parameter, g_max.

    Returns:
        List[float]: The sequence which represents Niggli reduced cell.
    """
    while True:
        niggli = [random.uniform(0, g_max) for _ in range(3)]
        niggli.sort()
        niggli.append(random.uniform(-0.5 * niggli[0], 0.5 * niggli[0]))
        niggli.append(random.uniform(0, 0.5 * niggli[0]))
        niggli.append(random.uniform(0, 0.5 * niggli[1]))

        if (niggli[0] + niggli[1] + 2 * niggli[3]) >= (2 * niggli[4] + 2 * niggli[5]):
            break

    return niggli
