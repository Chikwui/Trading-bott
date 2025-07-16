import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import settings
from database.models import Base

def init_db():
    """Initialize the database schema."""
    try:
        # Create engine and bind it to the metadata
        engine = create_engine(settings.DATABASE_URL)
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        print(f"Successfully initialized database at {settings.DATABASE_URL}")
        print("Tables created:", list(Base.metadata.tables.keys()))
        return True
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        if "sqlite" in str(settings.DATABASE_URL):
            print("\nSQLite-specific troubleshooting:")
            db_path = os.path.abspath("ai_trader.db")
            print(f"- Checking if database file exists at: {db_path}")
            if os.path.exists(db_path):
                print(f"  - Database file exists at: {db_path}")
                print(f"  - File size: {os.path.getsize(db_path)} bytes")
            else:
                print(f"  - Database file does not exist at: {db_path}")
                print("  - The file should be created automatically when the tables are created")
        return False

if __name__ == '__main__':
    if init_db():
        print("Database initialized successfully.")
    else:
        print("Failed to initialize database.")
        sys.exit(1)
