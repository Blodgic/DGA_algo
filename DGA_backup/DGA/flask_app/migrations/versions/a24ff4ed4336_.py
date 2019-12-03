"""empty message

Revision ID: a24ff4ed4336
Revises: ee37e599029f
Create Date: 2019-11-12 13:59:04.508939

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a24ff4ed4336'
down_revision = 'ee37e599029f'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('api_request_query_key', 'api_request', type_='unique')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint('api_request_query_key', 'api_request', ['query'])
    # ### end Alembic commands ###
