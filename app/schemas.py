# app/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class PrintArea(BaseModel):
    top_left: list[int] = Field(..., min_length=2, max_length=2)
    top_right: list[int] = Field(..., min_length=2, max_length=2)
    bottom_right: list[int] = Field(..., min_length=2, max_length=2)
    bottom_left: list[int] = Field(..., min_length=2, max_length=2)
    mask_points: Optional[list[list[int]]] = None


class LightingConfig(BaseModel):
    shadow_strength: float = 0.45
    highlight_strength: float = 0.55
    displacement_strength: float = 0.10
    specular_strength: float = 0.30
    specular_threshold: int = 220


class ColorConfig(BaseModel):
    enable_color_match: bool = True
    match_strength: float = 0.40


class EdgeConfig(BaseModel):
    feather_px: int = 6


class OutputConfig(BaseModel):
    jpeg_quality: int = 90


class TemplateConfig(BaseModel):
    print_area: PrintArea
    lighting: LightingConfig = LightingConfig()
    color: ColorConfig = ColorConfig()
    edge: EdgeConfig = EdgeConfig()
    output: OutputConfig = OutputConfig()


class TemplateCreate(BaseModel):
    slug: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    config: TemplateConfig
    output_width: int = 1500
    output_height: int = 1500


class TemplateConfigUpdate(BaseModel):
    config: TemplateConfig
    change_note: Optional[str] = None


class TemplateListItem(BaseModel):
    id: str
    slug: str
    name: str
    status: str
    preview_url: Optional[str] = None
    output_size: list[int]


class TemplateDetail(TemplateListItem):
    description: Optional[str] = None
    config: dict
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None
