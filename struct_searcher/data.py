import json
from pathlib import Path
from typing import Dict

PROCESSING_DIR_PATH = Path(__file__).resolve().parent.parent / "data" / "processing"


def load_atom_info() -> Dict[str, Dict[str, float]]:
    """Load atomic information

    Returns:
        Dict[str, Dict[str, float]]: Dict storing atomic information.
    """
    json_path = PROCESSING_DIR_PATH / "atom_info.json"
    with json_path.open("r") as f:
        atom_info = json.load(f)

    return atom_info
