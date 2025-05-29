from pymatgen.core import Lattice, Structure
from typing import Dict, Any


def decode_from_slices(slices_str: str) -> Dict[str, Any]:
    """
    Восстанавливает запись из SLICES-строки.
    Ожидает тот же формат, что и encode_to_slices.
    """
    parts = slices_str.split('|')
    material_id = parts[0]
    lat_vals = list(map(float, parts[1].split()))
    lattice = Lattice.from_parameters(*lat_vals)
    species = []
    coords = []
    for atom_line in parts[2:]:
        tokens = atom_line.split()
        sp = tokens[0]
        frac = list(map(float, tokens[1:4]))
        species.append(sp)
        coords.append(frac)
    struct = Structure(lattice, species, coords)
    return {'material_id': material_id, 'structure': struct.as_dict()}