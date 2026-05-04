import os
import subprocess
import yaml
from pathlib import Path
from dotenv import load_dotenv

def get_ext_repos_dir():
    load_dotenv(dotenv_path="config/paths.env")
    return os.getenv("EXT_REPOS_DIR", "ext_repos")

def clone_repo(repo_url, repo_name):
    ext_dir = Path(get_ext_repos_dir())
    ext_dir.mkdir(parents=True, exist_ok=True)
    target_path = ext_dir / repo_name
    
    if target_path.exists():
        print(f"[{repo_name}] Already cloned at {target_path}. Pulling latest...")
        subprocess.run(["git", "-C", str(target_path), "pull"], check=False)
    else:
        print(f"[{repo_name}] Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, str(target_path)], check=True)
    return target_path

