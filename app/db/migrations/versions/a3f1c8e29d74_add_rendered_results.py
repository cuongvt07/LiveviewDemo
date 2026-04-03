"""Add rendered_results table

Revision ID: a3f1c8e29d74
Revises: 8657ff5ebd58
Create Date: 2026-03-31 16:16:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a3f1c8e29d74'
down_revision: Union[str, Sequence[str], None] = '8657ff5ebd58'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create rendered_results table for fast URL → mockup resolution."""
    op.create_table(
        'rendered_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('url_slug', sa.String(length=500), nullable=False),
        sa.Column('image_path', sa.String(length=500), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_rendered_results_url_slug'),
        'rendered_results',
        ['url_slug'],
        unique=False,
    )
    op.create_index(
        op.f('ix_rendered_results_created_at'),
        'rendered_results',
        ['created_at'],
        unique=False,
    )


def downgrade() -> None:
    """Drop rendered_results table."""
    op.drop_index(op.f('ix_rendered_results_created_at'), table_name='rendered_results')
    op.drop_index(op.f('ix_rendered_results_url_slug'), table_name='rendered_results')
    op.drop_table('rendered_results')
