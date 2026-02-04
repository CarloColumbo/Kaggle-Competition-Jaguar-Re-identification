import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import subprocess


class Env:
    WANDB_API_KEY: str
    WANDB_PROJECT: str
    PROJECT_PATH: Path
    DATA_PATH: Path
    OUTPUT_PATH: Path
    CHECKPOINT_PATH: Path
    
    
    def __init__(self, experiment_name: str = "default_experiment"):
        if self._is_kaggle():
            self._load_kaggle_env()
        else:
            self._load_local_env()
            
        sys.path.insert(0, str(self.PROJECT_PATH))
            
        self.DATA_PATH = self.PROJECT_PATH / "data"
        self.OUTPUT_PATH = self.PROJECT_PATH / "output" / experiment_name
        self.CHECKPOINT_PATH = self.PROJECT_PATH / "checkpoints" / experiment_name
        self.EMBEDDINGS_PATH = self.CHECKPOINT_PATH / "embeddings"
        
        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

        print(f"Environment loaded. Project path: {self.PROJECT_PATH}")

    def _load_kaggle_env(self):
        from kaggle_secrets import UserSecretsClient # type: ignore

        user_secrets = UserSecretsClient()
        self.WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
        self.WANDB_PROJECT = user_secrets.get_secret("WANDB_PROJECT")
        
        github_token = user_secrets.get_secret("GITHUB_TOKEN")
        local_project_name = "project"
        
        self._clone_repo(github_token, local_project_name)
        
        self.PROJECT_PATH = Path(f"/kaggle/working/{local_project_name}")


    def _load_local_env(self):
        self.PROJECT_PATH = Path.cwd().parent

        load_dotenv(dotenv_path=self.PROJECT_PATH / ".env")
        
        self.WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        self.WANDB_PROJECT = os.getenv("WANDB_PROJECT")

        
    def _is_kaggle(self) -> bool:
        return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    
    def _clone_repo(self, token: str, dest_name: str):
        repo_url = f"https://{token}@github.com/CarloColumbo/CILP-Assessment-Multimodal-Learning.git"

        subprocess.run(
            ["git", "clone", repo_url, dest_name],
            check=True
        )