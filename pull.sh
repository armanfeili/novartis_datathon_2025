#!/bin/bash

# Git Pull Script
# Automates pulling latest changes from GitHub
# Usage: ./pull.sh [--branch <branch_name>]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
TARGET_BRANCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch|-b)
            TARGET_BRANCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./pull.sh [--branch <branch_name>]"
            echo "  --branch, -b    Specify target branch to pull from"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Starting Git Pull Process...${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${YELLOW}Current branch: ${CURRENT_BRANCH}${NC}"

# Fetch latest from remote
echo -e "${YELLOW}Fetching from remote...${NC}"
git fetch origin

if [ $? -ne 0 ]; then
    echo -e "${RED}Fetch failed. Check your internet connection.${NC}"
    exit 1
fi

# Check if there are local changes that need stashing
STASHED=false
if ! git diff-index --quiet HEAD -- 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo -e "${YELLOW}You have uncommitted changes. Stashing them...${NC}"
    git stash push -m "Auto-stash before pull operation"
    STASHED=true
fi

# If target branch specified and different from current, switch to it
if [ -n "$TARGET_BRANCH" ] && [ "$TARGET_BRANCH" != "$CURRENT_BRANCH" ]; then
    echo -e "${YELLOW}Target branch: ${TARGET_BRANCH}${NC}"
    
    # Check if target branch exists locally
    if git show-ref --verify --quiet refs/heads/"$TARGET_BRANCH"; then
        echo -e "${YELLOW}Switching to existing branch: ${TARGET_BRANCH}${NC}"
        git checkout "$TARGET_BRANCH"
    else
        # Check if it exists on remote
        if git show-ref --verify --quiet refs/remotes/origin/"$TARGET_BRANCH"; then
            echo -e "${YELLOW}Checking out remote branch: ${TARGET_BRANCH}${NC}"
            git checkout -b "$TARGET_BRANCH" origin/"$TARGET_BRANCH"
        else
            echo -e "${RED}Error: Branch '${TARGET_BRANCH}' does not exist locally or on remote${NC}"
            if [ "$STASHED" = true ]; then
                echo -e "${YELLOW}Restoring stashed changes...${NC}"
                git stash pop
            fi
            exit 1
        fi
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to switch to branch: ${TARGET_BRANCH}${NC}"
        if [ "$STASHED" = true ]; then
            git stash pop
        fi
        exit 1
    fi
    
    BRANCH="$TARGET_BRANCH"
else
    BRANCH="$CURRENT_BRANCH"
fi

echo -e "${YELLOW}Working on branch: ${BRANCH}${NC}"

# Pull latest changes
echo -e "${YELLOW}Pulling latest changes from origin/${BRANCH}...${NC}"
git pull origin "$BRANCH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled latest changes on branch: ${BRANCH}${NC}"
else
    echo -e "${RED}Pull failed. You may have merge conflicts to resolve.${NC}"
    if [ "$STASHED" = true ]; then
        echo -e "${YELLOW}Restoring stashed changes...${NC}"
        git stash pop
    fi
    exit 1
fi

# Restore stashed changes if any
if [ "$STASHED" = true ]; then
    echo -e "${YELLOW}Restoring stashed changes...${NC}"
    git stash pop
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Could not restore stashed changes automatically. Use 'git stash pop' manually.${NC}"
    fi
fi

# Show current status
echo -e "${YELLOW}Current status:${NC}"
git status --short

echo -e "${GREEN}Pull complete!${NC}"
