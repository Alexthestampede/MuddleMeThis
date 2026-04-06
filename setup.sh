#!/bin/bash
# MuddleMeThis Setup Script

echo "🎨 MuddleMeThis Setup"
echo "===================="
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install main requirements
echo "📥 Installing main application requirements..."
pip install -r requirements.txt

# Install DTgRPCconnector requirements
echo "📥 Installing DTgRPCconnector requirements..."
pip install -r dev/DTgRPCconnector/requirements.txt
pip install "flatbuffers>=24.3.0"  # Override for Python 3.13+ compatibility (aarch64 piwheels fix)

# Install ModuLLe
echo "📥 Installing ModuLLe..."
cd dev/ModuLLe
pip install -e .
cd ../..

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  1. Activate the venv: source venv/bin/activate"
echo "  2. Run the app: python app.py"
echo "  3. Open browser to: http://localhost:7860"
echo ""
echo "Configure your servers in the Settings tab:"
echo "  - LLM: http://192.168.2.20:1234"
echo "  - gRPC: 192.168.2.150:7859"
