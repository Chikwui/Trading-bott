import psycopg2
from config.settings import settings
import os

def setup_database():
    """
    Setup PostgreSQL database and user with proper authentication.
    This script should be run with PostgreSQL superuser credentials.
    """
    try:
        # Connect to PostgreSQL with superuser
        conn = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            password=os.getenv('POSTGRES_PASSWORD', ''),
            host='localhost',
            port='5432'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create the database if it doesn't exist
        cursor.execute("""
            SELECT 1 FROM pg_database WHERE datname = 'ai_trader'
        """)
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("""
                CREATE DATABASE ai_trader;
            """)
            print("Created database 'ai_trader'")
        
        # Create the trader user if it doesn't exist
        cursor.execute("""
            SELECT 1 FROM pg_roles WHERE rolname = 'trader'
        """)
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("""
                CREATE USER trader WITH PASSWORD 'secret';
                GRANT ALL PRIVILEGES ON DATABASE ai_trader TO trader;
            """)
            print("Created user 'trader' with password 'secret'")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("Database setup completed successfully")
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        raise

if __name__ == '__main__':
    setup_database()
