#!/bin/bash

# Git Push Script
# Automates staging, committing, and pushing changes to GitHub
# Usage: ./push.sh [--branch <branch_name>] [-m <commit_message>]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
TARGET_BRANCH=""
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch|-b)
            TARGET_BRANCH="$2"
            shift 2
            ;;
        --message|-m)
            COMMIT_MSG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./push.sh [--branch <branch_name>] [-m <commit_message>]"
            echo "  --branch, -b    Specify target branch to push to"
            echo "  --message, -m   Commit message"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            # If no flag, treat as commit message for backward compatibility
            if [ -z "$COMMIT_MSG" ]; then
                COMMIT_MSG="$1"
            fi
            shift
            ;;
    esac
done

echo -e "${YELLOW}Starting Git Push Process...${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# If target branch specified and different from current, switch to it
if [ -n "$TARGET_BRANCH" ] && [ "$TARGET_BRANCH" != "$CURRENT_BRANCH" ]; then
    echo -e "${YELLOW}Target branch: ${TARGET_BRANCH}${NC}"
    echo -e "${YELLOW}Current branch: ${CURRENT_BRANCH}${NC}"
    
    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD -- 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        echo -e "${YELLOW}Stashing current changes before switching branches...${NC}"
        git stash push -m "Auto-stash before switching to $TARGET_BRANCH"
        STASHED=true
    else
        STASHED=false
    fi
    
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
            echo -e "${YELLOW}Creating new branch: ${TARGET_BRANCH}${NC}"
            git checkout -b "$TARGET_BRANCH"
        fi
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to switch to branch: ${TARGET_BRANCH}${NC}"
        if [ "$STASHED" = true ]; then
            git stash pop
        fi
        exit 1
    fi
    
    # Restore stashed changes to new branch
    if [ "$STASHED" = true ]; then
        echo -e "${YELLOW}Restoring stashed changes to ${TARGET_BRANCH}...${NC}"
        git stash pop
    fi
    
    BRANCH="$TARGET_BRANCH"
else
    BRANCH="$CURRENT_BRANCH"
fi

echo -e "${YELLOW}Working on branch: ${BRANCH}${NC}"

# Show current status
echo -e "${YELLOW}Current Git Status:${NC}"
git status --short

# Check if there are any changes to commit
if git diff-index --quiet HEAD -- 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo -e "${YELLOW}No changes to commit${NC}"
    exit 0
fi

# Stage all changes
echo -e "${YELLOW}Staging all changes...${NC}"
git add -A

# Get commit message if not provided
if [ -z "$COMMIT_MSG" ]; then
    echo -e "${YELLOW}Enter commit message (or press Enter for default):${NC}"
    read -r COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Update $(date '+%Y-%m-%d %H:%M:%S')"
    fi
fi

# Commit changes
echo -e "${YELLOW}Committing with message: ${COMMIT_MSG}${NC}"
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo -e "${RED}Commit failed${NC}"
    exit 1
fi

# Push to remote
echo -e "${YELLOW}Pushing to origin/${BRANCH}...${NC}"
git push origin "$BRANCH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pushed to GitHub on branch: ${BRANCH}${NC}"
else
    echo -e "${RED}Push failed. You may need to pull first or resolve conflicts.${NC}"
    exit 1
fi
