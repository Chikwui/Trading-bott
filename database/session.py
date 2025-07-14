from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import settings

# Database engine
engine = create_engine(settings.DATABASE_URL)
# Session factory
SessionLocal = sessionmaker(bind=engine)
