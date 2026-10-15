# Push this project to GitHub

Open PowerShell in the extracted project folder and run:

```powershell
git init
git add .
git commit -m "Add smart parking occupancy detection project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your actual GitHub repository URL. If the repository already has commits or a remote, check `git status` and `git remote -v` before running these commands. Do not force-push unless you understand the consequences.

Then deploy through Streamlit Community Cloud using `streamlit_app.py` as the main file.
