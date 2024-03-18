import pydantic as _pydantic
from typing import Optional
from fastapi import UploadFile
from typing import List


class _PromptBase(_pydantic.BaseModel):
    seed: Optional[float] = -1
    num_inference_steps: int = 30
    guidance_scale: float = 12
    strength: float = 0.6


class PregnancyCreate(_PromptBase):
    encoded_base_img: List[str]
    img_height: int = 512
