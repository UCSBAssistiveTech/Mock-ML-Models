# GitHub Workflow Guide 

Follow these steps to add or update your mock ML model notebooks in this shared repository.

1. Go to the folder where you want the project to live
  cd ~/Documents  # or any directory you prefer

2. Download / Set up git if not yet installed:
  download git: https://git-scm.com/download/win
  git config --global user.name "Your Name"
  git config --global user.email "youremail@example.com"
  git --version

3. Clone the repo
  git clone https://github.com/UCSBAssistiveTech/Mock-ML-Models.git

4. Enter the repo folder
cd Mock-ML-Models

5. Pull up files
  code .

6. Add all new or changed files
  git add .

7. Or add specific files
  git add models/random_forest.ipynb

8. Commit Your Changes

Every time you make progress, commit with a clear message:

git commit -m "Add mock model for Random Forest"

9. Push to GitHub

When your work is ready to share:

git push origin main


# -- Common Commands Reference --
1. Check current branch:	git branch
2. View recent commits:	git log --oneline
3. Undo last commit (local only);	git reset --soft HEAD~1
4. Remove all staged files:	git reset
5. Check what changed:	git status
6. Pull latest main updates:	git pull origin main
