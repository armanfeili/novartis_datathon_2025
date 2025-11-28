#!/bin/bash

# Wrapper script - calls the main push.sh from parent directory
# Usage: ./push.sh [--branch <branch_name>] [-m <commit_message>]

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the parent push.sh with all arguments
bash "$SCRIPT_DIR/../push.sh" "$@"
