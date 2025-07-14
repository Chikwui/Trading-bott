# Setup Guide

This guide covers environment and infrastructure setup for AI Trader.

## Prerequisites
- **Python 3.10+**
- **Git** and a **GitHub** account
- **Docker** and **Docker Compose** (for PostgreSQL)
- **PostgreSQL** (production)
- **SQLite** (default local DB)

## Python Virtual Environment
```bash
python -m venv venv
# Windows
env\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

## SQLite (Local development)
No additional setup required. Default DB URL:
```
DATABASE_URL=sqlite:///ai_trader.db
```

## PostgreSQL (Production)
You have two setup options:

**A) Containerized (with Docker Compose)**
1. Start container:
   ```bash
docker-compose up -d postgres
# or
docker run -d --name ai-trader-postgres \
  -e POSTGRES_USER=trader -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 postgres:15
```

**B) Native installation (Windows or WSL2/Linux)**
1. Install PostgreSQL:
   - Windows: Download and run installer from https://www.postgresql.org/download/windows/
   - WSL2/macOS/Linux: `sudo apt update && sudo apt install -y postgresql`
2. Ensure service is running on port 5432.

After either option, update `.env`:
```ini
DATABASE_URL=postgresql://trader:secret@localhost:5432/ai_trader
```

Initialize schema:
```bash
python -m database.init_db
```

## Git & GitHub
1. Initialize repo:
   ```bash
git init
git add .
git commit -m "Initial commit: scaffold AI Trader"
```  
2. Create remote and push:
   ```bash
git remote add origin git@github.com:yourusername/ai-trader.git
git push -u origin main
```

