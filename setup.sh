#!/bin/bash

# RAG Demo Setup Script

echo "Setting up RAG Demo Environment"
echo "=================================="

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo -e "\nActivating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip

# Install requirements
echo -e "\nInstalling dependencies..."
pip install -r requirements.txt
