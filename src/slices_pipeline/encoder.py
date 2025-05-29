from pymatgen.core import Structure
from typing import Dict

def encode_to_slices(record: Dict) -> str:
    """
    Конвертирует запись в строку SLICES.
    Формат:
     - material_id
     - lattice: a b c alpha beta gamma
     - один атом на строку: symbol x_frac y_frac z_frac
    Части разделяются '|'.
    """
    struct = Structure.from_dict(record['structure'])
    lat    = struct.lattice
    parts  = []

    parts.append(record['material_id'])

    parts.append(
        f"{lat.a:.6f} {lat.b:.6f} {lat.c:.6f} "
        f"{lat.alpha:.6f} {lat.beta:.6f} {lat.gamma:.6f}"
    )

    for site in struct:
        x, y, z = site.frac_coords
        parts.append(f"{site.species_string} {x:.6f} {y:.6f} {z:.6f}")

    return "|".join(parts)