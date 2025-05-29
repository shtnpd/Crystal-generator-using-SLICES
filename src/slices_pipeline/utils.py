from typing import Dict, Any, Tuple
from pymatgen.core import Structure


def compare_structures(s1: Structure, s2: Structure, tol: float = 1e-6) -> Tuple[bool, Dict]:
    """
    Сравнивает две структуры:
     - lattice параметры в пределах tol
     - одинаковое количество сайтов
     - species и frac_coords каждого сайта в пределах tol
    Возвращает (all_equal, diffs).
    diffs содержит ключи с описанием расхождений.
    """
    diffs = {}
    for attr in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'):
        v1 = getattr(s1.lattice, attr)
        v2 = getattr(s2.lattice, attr)
        if abs(v1 - v2) > tol:
            diffs[f'lattice_{attr}'] = (v1, v2)
    if len(s1) != len(s2):
        diffs['n_sites'] = (len(s1), len(s2))
    for i, (site1, site2) in enumerate(zip(s1, s2)):
        if site1.species_string != site2.species_string:
            diffs[f'site_{i}_species'] = (site1.species_string, site2.species_string)
        for coord_idx, (c1, c2) in enumerate(zip(site1.frac_coords, site2.frac_coords)):
            if abs(c1 - c2) > tol:
                diffs[f'site_{i}_coord_{coord_idx}'] = (c1, c2)
    return (len(diffs) == 0), diffs


def compare_records(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сравнивает две записи: проверяет material_id и структуру.
    Возвращает метрики: all_equal и diffs.
    """
    result = {'all_equal': False, 'diffs': {}}
    if a.get('material_id') != b.get('material_id'):
        result['diffs']['material_id'] = (a.get('material_id'), b.get('material_id'))
        return result
    s1 = Structure.from_dict(a['structure'])
    s2 = Structure.from_dict(b['structure'])
    eq, struct_diffs = compare_structures(s1, s2)
    result['all_equal'] = eq
    result['diffs'] = struct_diffs
    return result