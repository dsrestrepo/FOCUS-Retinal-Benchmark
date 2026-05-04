import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def load_models_from_config(config_path="config/mllm_config.yaml"):
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get("models", [])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MLLM models")
    parser.add_argument("--config", type=str, default="config/mllm_config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    models_to_download = load_models_from_config(args.config)
    
    if not models_to_download:
        print("No models found to download in the configuration file.")
        sys.exit(0)

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in environment variables. Some models may require authentication.")

    print(f"Starting download for {len(models_to_download)} MLLM models from {args.config}...")
    
    for repo_id in models_to_download:
        print(f"\n--- Downloading {repo_id} ---")
        try:
            path = snapshot_download(repo_id=repo_id, token=token, repo_type="model")
            print(f"Successfully downloaded {repo_id} to {path}")
        except Exception as e:
            print(f"Failed to download {repo_id}. Error: {e}", file=sys.stderr)

    print("\nAll MLLM download tasks completed.")
