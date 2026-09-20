#!/usr/bin/env bash
# Lab 02 setup script — run this once inside your activated environment
# (e.g. after `conda activate ai-lab`).
#
# Usage:
#   bash setup.sh

set -e

echo "Installing Python packages..."
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading TextBlob corpora..."
python -m textblob.download_corpora

echo ""
echo "Done! Now run: jupyter notebook"
echo "Then open Lab_02_Web_Scraping_and_EDA.ipynb and Run All."
