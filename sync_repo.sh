#!/bin/bash

# Ensure we're on the main branch
git checkout main

# Fetch the latest changes from the remote
git fetch origin

# Check if there are any remote changes we don't have locally
if [ $(git rev-parse HEAD) != $(git rev-parse origin/main) ]; then
    echo "Warning: Your local main branch is different from the remote."
    echo "This script will overwrite the remote with your local changes."
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
    fi
fi

# Add all changes
git add .

# Commit changes if there are any
if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Reorganized folder structure"
fi

# Force push to remote
git push --force origin main

echo "Remote repository has been updated to match your local state."
