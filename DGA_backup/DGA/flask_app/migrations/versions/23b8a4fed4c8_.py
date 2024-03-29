"""empty message

Revision ID: 23b8a4fed4c8
Revises: dfb9d3cab77b
Create Date: 2019-11-12 00:08:57.731553

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '23b8a4fed4c8'
down_revision = 'dfb9d3cab77b'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('site_request_page_key', 'site_request', type_='unique')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint('site_request_page_key', 'site_request', ['page'])
    # ### end Alembic commands ###
