#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Color codes for logging
BLUE="\033[1;34m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; NC="\033[0m"
# Required Python version for the project
REQUIRED_PYTHON_VERSION="3.11"

# --- System Dependencies ---
echo -e "${BLUE}🔄  Updating apt cache…${NC}"
sudo apt-get update -y

echo -e "${BLUE}📥  Adding deadsnakes PPA for Python ${REQUIRED_PYTHON_VERSION}…${NC}"
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y

echo -e "${BLUE}🐍  Installing Python ${REQUIRED_PYTHON_VERSION} and required system tools…${NC}"
sudo apt-get install -y \
    "python${REQUIRED_PYTHON_VERSION}" \
    "python${REQUIRED_PYTHON_VERSION}-venv" \
    "python${REQUIRED_PYTHON_VERSION}-dev" \
    curl \
    poppler-utils \
    ghostscript \
    jq

echo -e "${GREEN}✅  Python check:${NC}"
printf "   • python%s → %s\n" "${REQUIRED_PYTHON_VERSION}" "$(python3.11 --version)"

# --- Poetry Installation & Setup ---
echo -e "${BLUE}📥  Installing Poetry (package manager)…${NC}"
curl -sSL https://install.python-poetry.org | "python${REQUIRED_PYTHON_VERSION}" -

# Add Poetry to the PATH for the current script session
export PATH="$HOME/.local/bin:$PATH"

echo -e "${GREEN}✅  Poetry check:${NC}"
printf "   • poetry → %s\n" "$(poetry --version)"

echo -e "${BLUE}🛠️   Configuring Poetry to use Python ${REQUIRED_PYTHON_VERSION}…${NC}"
poetry env use "python${REQUIRED_PYTHON_VERSION}"

# --- Project Dependency Installation ---
if [[ -f "pyproject.toml" ]]; then
    echo -e "${BLUE}📦  Installing project dependencies with Poetry…${NC}"
    # The 'poetry install' command creates the virtual environment and installs dependencies
    poetry install --no-root
else
    echo -e "${YELLOW}📄  No pyproject.toml found; skipping dependency installation.${NC}"
fi

# --- Cleanup ---
echo -e "${BLUE}🧹  Cleaning up unused apt packages…${NC}"
sudo apt-get autoremove -y

# --- Final Instructions ---
echo -e "${GREEN}🎉  Setup complete! Here's how to run your project:${NC}"
echo ""
echo -e "${BLUE}➡️  To run a single command (Recommended Method):${NC}"
echo -e "   Use ${YELLOW}poetry run${NC} before your command. You don't need to activate a shell."
echo -e "   ${GREEN}Example:${NC}  ${YELLOW}poetry run python run_app.py${NC}"
echo ""
echo -e "${BLUE}➡️  To open an interactive shell (for running multiple commands):${NC}"
echo -e "   The ${YELLOW}poetry shell${NC} command now requires a plugin (this is a one-time setup)."
echo -e "   ${GREEN}1. First, install the plugin:${NC} ${YELLOW}poetry self add poetry-plugin-shell${NC}"
echo -e "   ${GREEN}2. Then, you can use the command as usual:${NC} ${YELLOW}poetry shell${NC}"
echo ""
echo -e "${GREEN}We recommend using ${YELLOW}poetry run${NC} for most tasks as it's simple and reliable.${NC}"