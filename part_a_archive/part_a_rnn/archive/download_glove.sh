#!/bin/bash
# Download GloVe 50d embeddings
# This script downloads the GloVe embeddings needed for training

set -e  # Exit on error

echo "================================================"
echo "Downloading GloVe 50d Embeddings"
echo "================================================"
echo ""

# Set paths
GLOVE_DIR="/home/lab/rabanof/Emotion_Detection_DL/glove"
GLOVE_FILE="$GLOVE_DIR/glove.6B.50d.txt"

# Check if already downloaded
if [ -f "$GLOVE_FILE" ]; then
    echo "✓ GloVe 50d file already exists:"
    ls -lh "$GLOVE_FILE"
    echo ""
    echo "You're ready to run training!"
    echo "Run: cd /home/lab/rabanof/projects/Emotion_Detection_DL && ./run_on_gpu.sh"
    exit 0
fi

# Create directory
echo "Creating directory: $GLOVE_DIR"
mkdir -p "$GLOVE_DIR"
cd "$GLOVE_DIR"

# Download
echo ""
echo "Downloading GloVe embeddings (~860MB)..."
echo "This may take 2-5 minutes depending on connection speed."
echo ""
wget -c http://nlp.stanford.edu/data/glove.6B.zip

# Extract
echo ""
echo "Extracting glove.6B.50d.txt..."
unzip -o glove.6B.zip glove.6B.50d.txt

# Verify
if [ -f "$GLOVE_FILE" ]; then
    echo ""
    echo "✓ SUCCESS! GloVe 50d file downloaded:"
    ls -lh "$GLOVE_FILE"

    # Cleanup (optional - keep other dimensions commented out)
    # echo ""
    # echo "Cleaning up extra files..."
    # rm -f glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt

    echo ""
    echo "================================================"
    echo "Ready to run training!"
    echo "================================================"
    echo ""
    echo "Next steps:"
    echo "1. cd /home/lab/rabanof/projects/Emotion_Detection_DL"
    echo "2. ./run_on_gpu.sh"
    echo ""
else
    echo ""
    echo "✗ ERROR: Download failed"
    echo "Please check your internet connection and try again."
    exit 1
fi
