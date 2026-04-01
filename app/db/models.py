# app/db/models.py

import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    ForeignKey,
    DateTime,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Template(Base):
    __tablename__ = 'templates'

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        String(20),
        nullable=False,
        default='draft',
    )  # 'draft' | 'active' | 'archived'

    product_type = Column(String(20), nullable=False, default='mug')

    mockup_path     = Column(String(500), nullable=False)
    shadow_map_path = Column(String(500), nullable=True)
    normal_map_path = Column(String(500), nullable=True)
    specular_path   = Column(String(500), nullable=True)
    mask_path       = Column(String(500), nullable=False)

    config = Column(JSONB, nullable=False)

    output_width  = Column(Integer, nullable=False, default=1500)
    output_height = Column(Integer, nullable=False, default=1500)

    created_by = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    config_history = relationship(
        'TemplateConfigHistory',
        back_populates='template',
        order_by='TemplateConfigHistory.changed_at.desc()',
    )


class TemplateConfigHistory(Base):
    __tablename__ = 'template_config_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(
        UUID(as_uuid=True),
        ForeignKey('templates.id'),
        nullable=False,
    )
    config = Column(JSONB, nullable=False)
    changed_by = Column(String(100), nullable=True)
    changed_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    change_note = Column(Text, nullable=True)

    template = relationship(
        'Template',
        back_populates='config_history',
    )

class RenderedResult(Base):
    __tablename__ = 'rendered_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    url_slug = Column(String(500), nullable=False, index=True)
    image_path = Column(String(500), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )
