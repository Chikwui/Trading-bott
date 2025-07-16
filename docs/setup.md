# Setup Guide

This guide covers environment and infrastructure setup for AI Trader.

## Prerequisites

- **Python 3.10+** (includes Python 3.13 support; ensure `pip install --upgrade pip` and `pip install pyarrow>=18.0.0` for prebuilt wheels)
- **Git** and a **GitHub** account
- **Docker** and **Docker Compose** (for PostgreSQL)
- **PostgreSQL** (production)
- **SQLite** (default local DB)

## Python Virtual Environment
```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
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
  -p 5432:5432 postgres:17
```

**B) Native installation (Windows or WSL2/Linux)**
1. Install PostgreSQL:
   - Windows: Download and run installer for PostgreSQL 17 from https://www.postgresql.org/download/windows/
   - WSL2/macOS/Linux: `sudo apt update && sudo apt install -y postgresql`
2. Ensure service is running on port 5432.
3. Verify version: run `psql --version` (should show 17.x).

After either option, update `.env`:
```ini
DATABASE_URL=postgresql://trader:secret@localhost:5432/ai_trader
```

Initialize schema:
```bash
python -m database.init_db
```

## Kafka (Single-node KRaft Mode)

**Option A — Kafka 4.0 (no ZooKeeper)**  
1. Extract distribution:  
   ```powershell
   cd $Env:USERPROFILE\Downloads   # wherever the .tgz is
   tar -xf kafka_4.0.0_windows-x86_64.tgz -C C:\kafka
   ```  
   (or extract with 7-Zip)

2. Set environment variables (add to your PowerShell profile):  
   ```powershell
   $env:KAFKA_HOME = 'C:\kafka\kafka_4.0.0'
   $env:Path += ";$env:KAFKA_HOME\bin\windows"
   ```  
   *Tip: add to your `$PROFILE` for persistence.*

3. Configure single-node KRaft in `%KAFKA_HOME%\config\server.properties`:  
   ```ini
   node.id=1
   process.roles=broker,controller
   controller.quorum.voters=1@localhost:9093
   listeners=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093
   log.dirs=C:/kafka/data
   ```

4. Format metadata log (one-time):  
   ```powershell
   cd $Env:KAFKA_HOME
   $cid = [guid]::NewGuid().ToString()
   .\bin\windows\kafka-storage.bat format -t $cid -c .\config\server.properties
   ```

5. Start broker (keep window open):  
   ```powershell
   .\bin\windows\kafka-server-start.bat .\config\server.properties
   ```  
   *Wait for `INFO [main] started`.*

6. Smoke test:  
   ```powershell
   kafka-topics.bat --bootstrap-server localhost:9092 --create --topic ticks --partitions 1 --replication-factor 1
   echo "hello" | kafka-console-producer.bat --bootstrap-server localhost:9092 --topic ticks
   kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic ticks --from-beginning
   ```

⚠️ If any `zookeeper.connect` lines exist in `server.properties`, comment them out.

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

