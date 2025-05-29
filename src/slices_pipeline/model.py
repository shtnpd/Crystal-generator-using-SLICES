from pydantic import BaseModel
from typing import Dict, Any

class MaterialRecord(BaseModel):
    material_id: str
    structure: Dict[str, Any]