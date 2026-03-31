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
    specular_threshold: int = 180
    light_pos_x: float = 0.62
    light_pos_y: float = 0.32
    light_height: float = 55.0
    light_contrast: float = 50.0
    light_highlight: float = 60.0
    light_softness: float = 55.0
    cylinder_shading_strength: float = 0.28
    edge_darkening_strength: float = 0.12
    specular_line_strength: float = 0.65
    specular_line_position: float = 0.18
    specular_line_sigma: float = 0.12
    specular_line_blur_kernel: int = 11
    diffuse_highlight_strength: float = 0.25
    highlight_detail_strength: float = 0.35
    lighting_blur_kernel: int = 21
    highlight_blur_kernel: int = 9
    highlight_extract_blur_kernel: int = 41
    highlight_detail_blur_kernel: int = 9


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


class LibraryItem(BaseModel):
    name: str
    url: str
    size_bytes: int
    modified_at: str


class TemplateSaveAdhoc(BaseModel):
    slug: str
    name: str
    product_type: str = 'mug'
    config: dict  # JSON settings from editor
    mockup_url: str  # URL if library, or we might need to handle uploaded file separately
    # If it's a new upload, the frontend should upload it first to library, then call this.
    output_width: int = 1500
    output_height: int = 1500
    preview_data_url: Optional[str] = None


class UrlAnalysisImportRequest(BaseModel):
    source_url: str = Field(..., min_length=8, max_length=2000)
