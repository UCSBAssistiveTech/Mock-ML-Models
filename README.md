# GitHub Workflow Guide 

Follow these steps to add or update your mock ML model notebooks in this shared repository.

1. Go to the folder where you want the project to live:
- cd ~/Documents  # or any directory you prefer

3. Download / Set up git if not yet installed:
- download git: https://git-scm.com/download/win
- git config --global user.name "Your Name"
- git config --global user.email "youremail@example.com"
- git --version

4. Clone the repo
- git clone https://github.com/UCSBAssistiveTech/Mock-ML-Models.git

5. Enter the repo file
- cd Mock-ML-Models/models/[yourmodel].py

6. Pull up files
- code .

7. Add all new or changed files
- git add .

9. Commit Your Changes

- git commit -m "Add mock model for Random Forest"

9. Push to GitHub when your work is ready to share:

- git push origin main


# -- Common Commands Reference --
1. Check current branch:	git branch
2. View recent commits:	git log --oneline
3. Undo last commit (local only);	git reset --soft HEAD~1
4. Remove all staged files:	git reset
5. Check what changed:	git status
6. Pull latest main updates:	git pull origin main
