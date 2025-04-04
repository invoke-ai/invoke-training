#!/usr/bin/env bash

# --- Configuration ---
VENV_DIR=".venv"
PYTHON_CMD="python"
REQUIRE_PYTHON_VERSION="3.10"
TORCH_PACKAGE="torch==2.6.0+cu124"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
# --- End Configuration ---

# Exit immediately if a command exits with a non-zero status during critical sections.
# We'll enable/disable this as needed using 'set -e' and 'set +e'.

# --- Helper Functions ---

check_python() {
    echo "Checking for Python..."
    if ! command -v python &> /dev/null; then
        echo "python command not found, trying python3..."
        if ! command -v python3 &> /dev/null; then
            echo "Error: Neither python nor python3 command found."
            echo "Please install Python $REQUIRE_PYTHON_VERSION or higher and ensure it's in your PATH."
            exit 1
        fi
        PYTHON_CMD="python3"
    fi

    echo "Checking Python version..."
    # Use a subshell to avoid exiting the main script if the check fails
    (
        set -e
        $PYTHON_CMD -c "import sys; version = tuple(map(int, '$REQUIRE_PYTHON_VERSION'.split('.'))); assert sys.version_info >= version, f'Python $REQUIRE_PYTHON_VERSION or higher is required. You have Python {sys.version_info.major}.{sys.version_info.minor}.'"
    )
    if [ $? -ne 0 ]; then
        exit 1
    fi
    echo "Python check passed ($($PYTHON_CMD --version))."
}

install_app() {
    set -e # Enable exit on error for this function
    echo "--- Starting Installation ---"

    check_python

    if [ -d "$VENV_DIR" ]; then
        echo "Existing virtual environment found. Skipping creation."
    else
        echo "Creating Python virtual environment in $VENV_DIR..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi

    echo "Activating virtual environment for installation..."
    source "$VENV_DIR/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "This script defaults to installing PyTorch for CUDA 12.4 ($TORCH_PACKAGE)."
    read -p "Do you want to proceed with this version? (y/n, default: y): " confirm_torch
    confirm_torch=${confirm_torch:-y} # Default to yes if user presses Enter

    if [[ "$confirm_torch" =~ ^[Yy]$ ]]; then
        echo "Installing PyTorch ($TORCH_PACKAGE)..."
        pip install "$TORCH_PACKAGE" --extra-index-url "$TORCH_INDEX_URL"
    else
        echo "Skipping automatic PyTorch installation."
        echo "Please install PyTorch manually for your system using instructions from:"
        echo "https://pytorch.org/get-started/locally/"
        echo "Then, re-run this script's install option if needed (it will skip PyTorch if already installed)."
        # Optionally, you could offer to exit here or just continue installing other deps
        echo "Continuing to install other dependencies..."
        # The pip install . command should follow *outside* this if/else
    fi
    # The pip install . command comes after this block

    echo "Installing invoke-training and its dependencies..."
    # Assuming the script is run from the project root directory
    pip install .

    echo "Deactivating environment after installation..."
    deactivate

    echo "--- Installation complete! ---"
    set +e # Disable exit on error
}

run_app() {
    set -e # Enable exit on error for this function
    echo "--- Starting Application ---"
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "Error: Virtual environment not found at '$VENV_DIR'."
        echo "Please run the installation option first."
        return 1 # Use return instead of exit to go back to menu
    fi

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo "Starting the Invoke Training UI (invoke-train-ui)..."
    # Add any command-line arguments needed by invoke-train-ui here if necessary
    invoke-train-ui

    echo "Invoke Training UI finished. Deactivating environment..."
    deactivate
    echo "--- Application finished ---"
    set +e # Disable exit on error
}

reinstall_app() {
    echo "--- Reinstalling Application ---"
    if [ -d "$VENV_DIR" ]; then
        echo "Removing existing virtual environment '$VENV_DIR'..."
        rm -rf "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to remove existing virtual environment."
            return 1
        fi
        echo "Existing environment removed."
    else
        echo "No existing virtual environment found to remove."
    fi
    install_app
}

# --- Main Menu Logic ---

while true; do
    echo ""
    echo "-----------------------------------------"
    echo " Invoke Training Setup & Run Menu"
    echo "-----------------------------------------"
    echo " Virtual Environment Path: $VENV_DIR"
    if [ -d "$VENV_DIR" ]; then
        echo " Status: Virtual environment detected."
        options=("Run Invoke Training UI" "Reinstall (Deletes and rebuilds $VENV_DIR)" "Exit")
        actions=(run_app reinstall_app exit_script)
    else
        echo " Status: Virtual environment NOT detected."
        options=("Install Invoke Training" "Exit")
        actions=(install_app exit_script)
    fi

    PS3="Please enter your choice: "
    select opt in "${options[@]}"; do
        # Find the index of the selected option
        index=-1
        for i in "${!options[@]}"; do
            if [[ "${options[$i]}" = "$opt" ]]; then
                index=$i
                break
            fi
        done

        if [[ "$index" != -1 ]]; then
            action_to_run="${actions[$index]}"
            if [[ "$action_to_run" == "exit_script" ]]; then
                echo "Exiting."
                exit 0
            else
                "$action_to_run" # Execute the chosen function
                break # Break select loop, go back to main while loop
            fi
        else
            echo "Invalid option $REPLY"
        fi
    done
    echo "-----------------------------------------"
    read -p "Press Enter to return to the menu..." # Pause before showing menu again
done

exit 0 # Should not be reached, but good practice 