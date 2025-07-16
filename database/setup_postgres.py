import os
import sys
from pathlib import Path
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings

def get_db_connection(dbname="postgres", use_postgres_user=False, password=None):
    """Get a database connection with the specified database name."""
    # Parse the DATABASE_URL
    from urllib.parse import urlparse
    db_url = settings.DATABASE_URL
    result = urlparse(db_url)
    
    # Get connection parameters
    params = {
        'dbname': dbname,
        'host': result.hostname or 'localhost',
        'port': result.port or 5432,
        'connect_timeout': 5  # Add a timeout to avoid hanging
    }
    
    if use_postgres_user:
        # Try to connect as postgres superuser
        if password is None:
            try:
                import getpass
                password = getpass.getpass("Enter PostgreSQL 'postgres' user password: ")
            except Exception:
                print("Warning: Could not prompt for password. Using default 'postgres' password.")
                password = 'postgres'
        
        params.update({
            'user': 'postgres',
            'password': password
        })
    else:
        # Use credentials from DATABASE_URL
        params.update({
            'user': result.username or 'postgres',
            'password': result.password or 'postgres'
        })
    
    try:
        conn = psycopg2.connect(**params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def check_or_create_user():
    """Check if the 'trader' user exists, create if it doesn't."""
    # First try with postgres superuser
    password = None
    for attempt in range(3):  # Give 3 attempts to enter the correct password
        conn = get_db_connection(use_postgres_user=True, password=password)
        if conn:
            break
        if attempt < 2:  # Don't show this on the last attempt
            print("Authentication failed. Please try again.")
            password = None  # Reset to prompt again
    
    if not conn:
        # If that fails, try with credentials from DATABASE_URL as a last resort
        print("\nTrying with credentials from DATABASE_URL...")
        conn = get_db_connection()
        if not conn:
            print("\nFailed to connect to PostgreSQL server. Please check:")
            print("1. Is PostgreSQL running on port 5432?")
            print("2. Is the 'postgres' user password correct?")
            print("3. Check your DATABASE_URL in .env file")
            return False
        
    try:
        with conn.cursor() as cur:
            # Check if user exists
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'trader'")
            if not cur.fetchone():
                # Create user with password from .env
                db_url = settings.DATABASE_URL
                password = db_url.split(':')[2].split('@')[0]  # Extract password from URL
                create_user_sql = sql.SQL("CREATE USER trader WITH PASSWORD %s")
                cur.execute(create_user_sql, (password,))
                print("Created 'trader' user")
            else:
                print("'trader' user already exists")
        return True
    except Exception as e:
        print(f"Error checking/creating user: {e}")
        return False
    finally:
        conn.close()

def check_or_create_database():
    """Check if the database exists, create if it doesn't."""
    # Get database name from URL
    db_url = settings.DATABASE_URL
    dbname = db_url.split('/')[-1].split('?')[0]  # Extract database name
    
    # First try with postgres superuser
    conn = get_db_connection(use_postgres_user=True)
    if not conn:
        # If that fails, try with credentials from DATABASE_URL
        conn = get_db_connection()
        if not conn:
            print("Failed to connect to PostgreSQL server. Is it running?")
            return False
        
    try:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if not cur.fetchone():
                # Create database
                create_db_sql = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
                cur.execute(create_db_sql)
                print(f"Created database '{dbname}'")
            else:
                print(f"Database '{dbname}' already exists")
        return True
    except Exception as e:
        print(f"Error checking/creating database: {e}")
        return False
    finally:
        conn.close()

def grant_privileges():
    """Grant all privileges on the database to the trader user."""
    # Get database name from URL
    db_url = settings.DATABASE_URL
    dbname = db_url.split('/')[-1].split('?')[0]  # Extract database name
    
    conn = get_db_connection(dbname)
    if not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            # Grant all privileges on database to trader user
            grant_sql = sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO trader").format(
                sql.Identifier(dbname)
            )
            cur.execute(grant_sql)
            
            # Grant all privileges on all tables in the database
            cur.execute("""
                GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
                GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader;
                GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO trader;
            """)
            
            # Set default privileges for future objects
            cur.execute("""
                ALTER DEFAULT PRIVILEGES IN SCHEMA public 
                GRANT ALL PRIVILEGES ON TABLES TO trader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public 
                GRANT ALL PRIVILEGES ON SEQUENCES TO trader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public 
                GRANT ALL PRIVILEGES ON FUNCTIONS TO trader;
            """)
            
            print(f"Granted all privileges on database '{dbname}' to 'trader' user")
        return True
    except Exception as e:
        print(f"Error granting privileges: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    print("=== Setting up PostgreSQL database and user ===")
    
    # First, check if psycopg2 is installed
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 is not installed. Please install it with:")
        print("pip install psycopg2-binary")
        sys.exit(1)
    
    # Check and create user
    if not check_or_create_user():
        print("Failed to set up database user. Please check PostgreSQL is running and accessible.")
        sys.exit(1)
    
    # Check and create database
    if not check_or_create_database():
        print("Failed to set up database. Please check PostgreSQL is running and accessible.")
        sys.exit(1)
    
    # Grant privileges
    if not grant_privileges():
        print("Failed to grant privileges. The database may not be fully functional.")
        sys.exit(1)
    
    print("\nPostgreSQL setup completed successfully!")
    print("You can now run 'python database/init_db.py' to initialize the database schema.")
