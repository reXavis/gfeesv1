#!/bin/bash

echo "🛑 Stopping GFeesV1 scripts..."

# Function to stop script by PID file
stop_script() {
    local script_name=$1
    local pid_file="${script_name%.py}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "⏹️  Stopping $script_name (PID: $pid)..."
            kill "$pid"
            sleep 2
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "🔨 Force stopping $script_name..."
                kill -9 "$pid"
            fi
            
            echo "✅ $script_name stopped"
        else
            echo "ℹ️  $script_name was not running"
        fi
        rm -f "$pid_file"
    else
        echo "ℹ️  No PID file found for $script_name"
    fi
}

# Stop both ingestor scripts
stop_script "dex_ingestor.py"
stop_script "gliquid_ingestor.py"
stop_script "fee_runner.py"

# Also kill any remaining processes by name (fallback)
echo "🧹 Cleaning up any remaining processes..."
pkill -f "dex_ingestor.py" 2>/dev/null || true
pkill -f "gliquid_ingestor.py" 2>/dev/null || true
pkill -f "fee_runner.py" 2>/dev/null || true
echo "✅ All scripts stopped successfully!"
