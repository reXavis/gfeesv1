#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/dex
mkdir -p data/gliquid
mkdir -p configs

# Check if config files exist
if [ ! -f "configs/ingestor_config.json" ]; then
    echo "⚠️  Warning: configs/ingestor_config.json not found"
    echo "   Please create this file with your DEX configuration"
fi

if [ ! -f "configs/gliquid_config.json" ]; then
    echo "⚠️  Warning: configs/gliquid_config.json not found"
    echo "   Please create this file with your GLiquid configuration"
fi

echo "🎯 Starting data ingestion scripts..."

# Function to run script in background and capture PID
run_script() {
    local script_name=$1
    local config_file=$2
    
    if [ -f "$config_file" ]; then
        echo "▶️  Starting $script_name..."
        python3 "$script_name" &
        local pid=$!
        echo "$pid" > "${script_name%.py}.pid"
        echo "   PID: $pid"
    else
        echo "⏭️  Skipping $script_name (config file not found: $config_file)"
    fi
}

# Start both ingestor scripts
run_script "dex_ingestor.py" "configs/ingestor_config.json"
run_script "gliquid_ingestor.py" "configs/gliquid_config.json"

echo ""
echo "✅ Setup complete! Both ingestor scripts are running in the background."
echo ""
echo "📊 Data will be saved to:"
echo "   - DEX data: data/dex/"
echo "   - GLiquid data: data/gliquid/"
echo ""
echo "🛑 To stop the scripts, run:"
echo "   ./stop.sh"
echo ""
echo "📋 To check running processes:"
echo "   ps aux | grep -E '(dex_ingestor|gliquid_ingestor)'"
echo ""
echo "📝 To view logs, check the terminal output or redirect to files:"
echo "   python3 dex_ingestor.py > logs/dex.log 2>&1 &"
echo "   python3 gliquid_ingestor.py > logs/gliquid.log 2>&1 &"