@echo off
REM start_kafka.bat - Automate Kafka KRaft start, topic creation, and console consumer

REM Generate a new cluster ID
for /f "delims=" %%i in ('powershell -NoProfile -Command "[guid]::NewGuid().Guid"') do set CLUSTER_ID=%%i

echo Cluster ID: %CLUSTER_ID%

if exist "C:\kafka\kraft-data" (
    echo Deleting existing data directory...
    rmdir /S /Q "C:\kafka\kraft-data"
)
echo Formatting storage directory...
call "C:\Kafka\kafka_2.13-4.0.0\bin\windows\kafka-storage.bat" format ^
  --config "C:\Kafka\kafka_2.13-4.0.0\config\kraft\server.properties" ^
  --cluster-id %CLUSTER_ID%

REM Start Kafka broker in a new window
echo Starting Kafka broker...
start "Kafka Broker" cmd /k ""C:\Kafka\kafka_2.13-4.0.0\bin\windows\kafka-server-start.bat" "C:\Kafka\kafka_2.13-4.0.0\config\kraft\server.properties""

REM Create test-topic if it doesn't exist
echo Creating test-topic...
powershell -NoProfile -Command "& 'C:\Kafka\kafka_2.13-4.0.0\bin\windows\kafka-topics.bat' --bootstrap-server localhost:9092 --create --topic test-topic --partitions 1 --replication-factor 1" || echo Topic may already exist.

REM Start console consumer in a new window
echo Starting console consumer...
start "Kafka Consumer" cmd /k ""C:\Kafka\kafka_2.13-4.0.0\bin\windows\kafka-console-consumer.bat" --bootstrap-server localhost:9092 --topic test-topic --from-beginning""

echo Done. Check the Kafka Broker and Consumer windows.
