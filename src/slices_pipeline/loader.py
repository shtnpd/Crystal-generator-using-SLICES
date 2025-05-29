import pandas as pd
from io import StringIO
from pathlib import Path
from typing import List, Dict

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

BASE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "MP20"
CSV_SPLITS = {
    "train": BASE / "train.csv",
    "val":   BASE / "val.csv",
    "test":  BASE / "test.csv",
}

def read_split_df(split: str) -> pd.DataFrame:
    return _read_split_df(split)

def _read_split_df(split: str) -> pd.DataFrame:
    path = CSV_SPLITS.get(split)
    if path is None:
        raise ValueError(f"Unknown split '{split}'")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "material_id" not in df.columns or "cif" not in df.columns:
        raise ValueError(f"CSV {path} must contain columns 'material_id' and 'cif'")
    return df

def _parse_structure_from_cif(cif_text: str) -> Structure:
    parser = CifParser(StringIO(cif_text))
    structs = parser.parse_structures(primitive=False)
    if not structs:
        raise ValueError("Не получилось распарсить CIF-текст")
    return structs[0]

def load_dataset(split: str) -> List[Dict]:
    df = _read_split_df(split)
    records = []
    for _, row in df.iterrows():
        struct = _parse_structure_from_cif(row["cif"])
        records.append({
            "material_id": row["material_id"],
            "structure": struct.as_dict()
        })
    return records

def load_mp20_record(material_id: str, split: str = "train") -> Dict:
    df = _read_split_df(split)
    sub = df[df["material_id"] == material_id]
    if sub.empty:
        raise KeyError(f"{material_id} not found in split '{split}'")
    row = sub.iloc[0]
    struct = _parse_structure_from_cif(row["cif"])
    return {
        "material_id": material_id,
        "structure": struct.as_dict()
    }
