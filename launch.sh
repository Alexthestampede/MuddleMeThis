#!/bin/bash
# MuddleMeThis Launch Script
# Simple launcher for the MuddleMeThis application

echo "🎨 Starting MuddleMeThis..."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    # Check if venv exists
    if [ -d "venv" ]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        echo "📦 Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

# Check if required modules are installed
python3 -c "import gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required dependencies not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Launch the application
echo "🚀 Launching MuddleMeThis..."
echo "📱 Access at: http://localhost:7860"
echo "📱 Network access: http://$(hostname -I | awk '{print $1}'):7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
