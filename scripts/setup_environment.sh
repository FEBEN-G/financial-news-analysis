#!/bin/bash

# Script to set up the development environment for financial news analysis

echo "Setting up financial news analysis environment..."

# Check if we're in the project directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install system dependencies for TA-Lib
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential python3-dev

# Try to install TA-Lib system library
echo "Installing TA-Lib system library..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# If TA-Lib installation failed, install a alternative
if ! python -c "import talib" 2>/dev/null; then
    echo "TA-Lib installation may have failed, installing fallback packages..."
    pip install pandas-ta
fi

echo "Environment setup complete!"
echo "To activate the virtual environment in the future, run: source venv/bin/activate"