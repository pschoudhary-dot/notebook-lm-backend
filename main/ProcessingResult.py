from typing import List, Union, Dict, Any
from pydantic import BaseModel

class ProcessingResult(BaseModel):
    success: bool
    content: str
    metadata: Dict[str, Any] = {}
    error_message: str = ""