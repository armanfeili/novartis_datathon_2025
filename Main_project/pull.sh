#!/bin/bash

# Wrapper script - calls the main pull.sh from parent directory
# Usage: ./pull.sh [--branch <branch_name>]

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the parent pull.sh with all arguments
bash "$SCRIPT_DIR/../pull.sh" "$@"
