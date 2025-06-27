#!/bin/bash

# ==============================================================================
#  Requirements Analysis Tool Starter (Bash-Only Edition)
#  - Starts, stops, and monitors the main app and demo server.
#  - Usage:
#      bash start.sh         (to start services)
#      bash start.sh monitor (to view live logs for both services)
#      bash start.sh stop    (to stop services and clean up logs)
# ==============================================================================

# --- Configuration ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Script Logic ---
# Assumes the script is run from the project's root directory
PROJECT_ROOT=$(pwd)
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
APP_PID_FILE="$PROJECT_ROOT/app.pid"
DEMO_PID_FILE="$PROJECT_ROOT/demo.pid"
APP_LOG_FILE="$PROJECT_ROOT/app.log"
DEMO_LOG_FILE="$PROJECT_ROOT/demo.log"

# --- Stop Functionality ---
if [ "$1" == "stop" ]; then
    echo -e "${YELLOW}🛑 Stopping Requirements Analysis Tool services...${NC}"
    # Stop main app if its PID file exists
    if [ -f "$APP_PID_FILE" ]; then
        echo "   - Stopping main app (PID: $(cat $APP_PID_FILE))..."
        kill $(cat $APP_PID_FILE)
        rm "$APP_PID_FILE"
    else
        echo "   - Main app not running via this script (no PID file)."
    fi

    # Stop judge demo if its PID file exists
    if [ -f "$DEMO_PID_FILE" ]; then
        echo "   - Stopping judge demo (PID: $(cat $DEMO_PID_FILE))..."
        kill $(cat $DEMO_PID_FILE)
        rm "$DEMO_PID_FILE"
    else
        echo "   - Judge demo not running via this script (no PID file)."
    fi

    # Clean up temporary log files
    echo "   - Cleaning up temporary log files..."
    rm -f "$APP_LOG_FILE" "$DEMO_LOG_FILE"

    echo -e "${GREEN}✅ Services stopped.${NC}"
    exit 0
fi

# --- Monitor Functionality ---
if [ "$1" == "monitor" ]; then
    echo -e "${BLUE}👀 Monitoring application logs... (Press Ctrl+C to exit)${NC}"
    if [ ! -f "$APP_LOG_FILE" ] && [ ! -f "$DEMO_LOG_FILE" ]; then
        echo -e "${YELLOW}Log streams not found. Start the services first with 'bash start.sh'.${NC}"
        exit 1
    fi

    # Clean up background tail processes on script exit
    trap 'echo -e "\n${YELLOW}🛑 Stopping log monitor...${NC}"; kill $(jobs -p) 2>/dev/null' EXIT

    # Tail main app logs with a purple prefix
    if [ -f "$APP_LOG_FILE" ]; then
        tail -n 100 -f "$APP_LOG_FILE" | sed "s/^/${PURPLE}[MAIN_APP]${NC} /" &
    fi

    # Tail demo app logs with a cyan prefix
    if [ -f "$DEMO_LOG_FILE" ]; then
        tail -n 100 -f "$DEMO_LOG_FILE" | sed "s/^/${CYAN}[DEMO_APP]${NC} /" &
    fi

    # Wait for background processes, allowing user to view logs
    wait
    exit 0
fi


# --- Header for Starting ---
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Starting Requirements Analysis Tool   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo "" # Newline for spacing

# --- Pre-run Checks ---
if [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}Error: 'pyproject.toml' not found."
    echo "Please run this script from the project's root directory."
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo -e "${YELLOW}Error: Python virtual environment not found at ${VENV_PATH}"
    echo "Please create it first (e.g., python3.11 -m venv .venv)."
    exit 1
fi

echo "🚀 Launching services in the background..."

# --- Environment and Dependency Setup ---
echo "   - Activating Python virtual environment..."
source "$VENV_PATH"

# Check for poetry and install if it's missing
if ! command -v poetry &> /dev/null
then
    echo -e "${YELLOW}   - 'poetry' command not found. Installing version 2.1.3 now...${NC}"
    # Use the official installer to get a specific version
    curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3
    
    # The installer may place the binary in ~/.local/bin, which needs to be added to the PATH
    # for the current script session.
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v poetry &> /dev/null
    then
        echo -e "${YELLOW}   - Poetry installation failed. Please check the output above.${NC}"
        exit 1
    fi
    echo -e "${GREEN}   - Poetry installed successfully.${NC}"
fi

# Ensure poetry uses the correct python version, as per the project's requirements
echo "   - Configuring poetry to use Python 3.11..."
poetry env use python3.11

# --- Service Launch ---

# Start Main Application as a background process
echo "   - Launching Main Application..."
poetry run python3 -u run_app.py > "$APP_LOG_FILE" 2>&1 &
echo $! > "$APP_PID_FILE"

# Start Judge Demo as a background process
echo "   - Launching Judge Demo..."
poetry run python3 -u run_judge_demo.py > "$DEMO_LOG_FILE" 2>&1 &
echo $! > "$DEMO_PID_FILE"

# Deactivate the virtual environment for the current shell session
deactivate

# --- Final Instructions ---
echo -e "\n${GREEN}✅ Services are launching in the background.${NC}"
echo -e "   - Main App (API on 8000, UI on 8501) is starting."
echo -e "   - Judge Demo (UI on 8502) is starting."
echo -e "\nAccess the interfaces at:"
echo -e "   - Main UI:     ${YELLOW}http://localhost:8501${NC}"
echo -e "   - Judge Demo:  ${YELLOW}http://localhost:8502${NC}"
echo -e "\nTo view live logs for both services, run this command:"
echo -e "${YELLOW}bash start.sh monitor${NC}"
echo -e "\nTo stop all services and clean up logs, run this command:"
echo -e "${YELLOW}bash start.sh stop${NC}\n"