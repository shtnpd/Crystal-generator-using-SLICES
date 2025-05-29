from typing import Dict, Any
from .loader   import load_mp20_record, load_dataset
from .encoder  import encode_to_slices
from .decoder  import decode_from_slices
from .utils    import compare_records

def round_trip(material_id: str, split: str = "train") -> Dict[str, Any]:
    rec    = load_mp20_record(material_id, split)
    slices = encode_to_slices(rec)
    rec2   = decode_from_slices(slices)
    metrics= compare_records(rec, rec2)
    return {
        "success": len(metrics["diffs"]) == 0,
        "metrics": metrics
    }
