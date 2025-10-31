#!/bin/bash

# RAG Demo Setup Script

echo "🚀 Setting up RAG Demo Environment"
echo "=================================="

# Create virtual environment
echo -e "\n📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo -e "\n✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\n⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo -e "\n📥 Installing dependencies..."
pip install -r requirements.txt

echo -e "\n✅ Setup complete!"
echo -e "\n📝 Next steps:"
echo "1. Set your Anthropic API key:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""
echo "2. Run the demo:"
echo "   python rag_demo.py"
echo ""
echo "3. To deactivate the virtual environment later:"
echo "   deactivate"
