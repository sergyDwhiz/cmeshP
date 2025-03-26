#!/bin/bash
set -e

# ======================
# 1. Environment Validation
# ======================
if [ "$(uname)" != "Darwin" ]; then
    echo "This script is for macOS. Please use the appropriate version for your OS."
    exit 1
fi

# ======================
# 2. Homebrew Setup
# ======================
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
        echo "Failed to install Homebrew. Please install manually from https://brew.sh"
        exit 1
    }
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# ======================
# 3. Setup Installation Directory
# ======================
echo "Creating FreeSurfer installation directory..."
sudo mkdir -p /usr/local/freesurfer

# ======================
# 4. Download FreeSurfer Archive
# ======================
echo "Downloading FreeSurfer..."
curl -L -o freesurfer-darwin-macOS-7.4.1.tar.gz \
    https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-darwin-macOS-7.4.1.tar.gz || {
    echo "Download failed. Please check your internet connection and try again."
    exit 1
}

# Verify the download
if [ ! -f "freesurfer-darwin-macOS-7.4.1.tar.gz" ]; then
    echo "Download verification failed. File not found."
    exit 1
}
filesize=$(stat -f%z "freesurfer-darwin-macOS-7.4.1.tar.gz")
if [ "$filesize" -lt 100000000 ]; then
    echo "Downloaded file is too small. Likely an error page. Aborting."
    rm "freesurfer-darwin-macOS-7.4.1.tar.gz"
    exit 1
}

# ======================
# 5. Extract Archive
# ======================
echo "Extracting FreeSurfer..."
sudo tar -xzf freesurfer-darwin-macOS-7.4.1.tar.gz -C /usr/local || {
    echo "Extraction failed."
    exit 1
}

# ======================
# 6. Configure Environment Variables
# ======================
echo "Setting up environment variables..."
grep -q "FREESURFER_HOME" ~/.zshrc || echo 'export FREESURFER_HOME=/usr/local/freesurfer' >> ~/.zshrc
grep -q "SetUpFreeSurfer.sh" ~/.zshrc || echo 'source $FREESURFER_HOME/SetUpFreeSurfer.sh' >> ~/.zshrc

# ======================
# 7. Cleanup and Completion
# ======================
rm freesurfer-darwin-macOS-7.4.1.tar.gz

echo "FreeSurfer installation completed!"
echo "Please restart your terminal or run: source ~/.zshrc"
echo "Note: You need to obtain a license file from: https://surfer.nmr.mgh.harvard.edu/registration.html"
