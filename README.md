# GitHub Workflow Guide 

Follow these steps to add or update your mock ML model notebooks in this shared repository.

1. Clone the Repository (First Time Only):

If you haven’t downloaded the repo yet:

# Go to the folder where you want the project to live
cd ~/Documents  # or any directory you prefer

# Download / Set up git:
download git: https://git-scm.com/download/win
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
git --version

# Clone the repo
git clone https://github.com/UCSBAssistiveTech/Mock-ML-Models.git

# Enter the repo folder
cd Mock-ML-Models

# Pull up files
code .

# Add all new or changed files
git add .

# Or add specific files
git add models/random_forest.ipynb

# Commit Your Changes

Every time you make progress, commit with a clear message:

ex) git commit -m "Add mock model for Random Forest"

# Push to GitHub

When your work is ready to share:

git push origin main

# END

# -- Common Commands Reference --
Check current branch:	git branch
View recent commits:	git log --oneline
Undo last commit (local only);	git reset --soft HEAD~1
Remove all staged files:	git reset
Check what changed:	git status
Pull latest main updates:	git pull origin main
