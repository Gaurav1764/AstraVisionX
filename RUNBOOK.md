# Kill old server on port 8000
$pid = (netstat -ano | findstr ":8000" | Select-String "LISTENING" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1); if ($pid) { taskkill /PID $pid /F }

# Start fresh
cd d:\Gaurav\AstraVisionX
.venv\Scripts\python app.py