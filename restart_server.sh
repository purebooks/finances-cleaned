#!/bin/bash
# A reliable script to find, stop, and restart the Flask server.

# --- Find the Process ID (PID) using the port ---
PID=$(lsof -t -i:8080)

if [ -z "$PID" ]
then
      echo "✅ No server found on port 8080. Starting new server..."
else
      echo "🔥 Server found with PID: $PID. Stopping it now..."
      # --- Forcefully stop the old process ---
      kill -9 $PID
      # --- Wait for the OS to release the port ---
      sleep 2 
      echo "🛑 Server stopped."
fi

# --- Start the new server with the latest code ---
echo "🚀 Starting new server..."
python3 app_v5.py
