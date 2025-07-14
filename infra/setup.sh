#!/usr/bin/env bash

# Infra setup for AI Trader on MCP servers
# Usage: ./infra/setup.sh
# Note: This bash script is for Linux/macOS. Windows users without WSL/Docker should run the PowerShell script: infra\setup.ps1 or follow manual steps in docs/setup.md

set -e

# 1. Check Docker and Docker Compose for containerized Postgres
SKIP_DOCKER=false
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
  echo "Docker or docker-compose not found. Skipping containerized PostgreSQL setup."
  echo "Ensure PostgreSQL is installed natively (Windows installer, WSL2 apt install, or use a cloud DB)."
  SKIP_DOCKER=true
fi

if [ "$SKIP_DOCKER" = false ]; then
  echo "Starting PostgreSQL container..."
  docker-compose up -d postgres
else
  echo "Assuming PostgreSQL is installed and running locally. Proceeding..."
fi

# Version check for native PostgreSQL (skip if containerized)
if [ "$SKIP_DOCKER" = true ]; then
  echo "Verifying local PostgreSQL version..."
  if ! command -v psql &> /dev/null; then
    echo "psql not found. Please install PostgreSQL 17 and ensure psql is in PATH." >&2
    exit 1
  fi
  VER=$(psql --version)
  MAJOR=$(echo "$VER" | grep -oE '[0-9]+' | head -1)
  if [ "$MAJOR" != "17" ]; then
    echo "Warning: found $VER, expected PostgreSQL 17.x" >&2
  else
    echo "PostgreSQL version $VER verified."
  fi
fi

# 3. Initialize Python environment
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# 4. Copy and edit environment variables
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Please update .env with your credentials."
fi

# 5. Initialize database schema
echo "Initializing database..."
python -m database.init_db

# 6. Git initialization (if needed)
if [ ! -d .git ]; then
  git init
  git add .
  git commit -m "Initial infra setup"
  echo "Add remote with: git remote add origin <repo-url>"
fi

echo "Infrastructure setup complete."
