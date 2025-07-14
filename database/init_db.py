from sqlalchemy import create_engine, MetaData
from config.settings import settings


def init_db():
    """Initialize the database schema."""
    engine = create_engine(settings.DATABASE_URL)
    meta = MetaData()
    # Import models here to create tables
    meta.create_all(engine)


if __name__ == '__main__':
    init_db()
    print("Database initialized.")
