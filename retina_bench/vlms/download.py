import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

VLM_HF_MODELS = {
    "medsiglip": "google/medsiglip-448",
    "siglip2": "google/siglip2-base-patch16-224",
    "clip-vit": "openai/clip-vit-base-patch32",
    "FLAIR": "jusiro2/FLAIR"
}

VLM_GDRIVE_MODELS = {
    "EyeCLIP": {
        "id": "1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr",
        "path": "ext_repos/EyeCLIP/eyeclip_visual.pt"
    },
    "RET-CLIP": {
        "id": "1lYrAg5qzFbNghEW-3UB36v9WL-mo5eN9",
        "path": "ext_repos/RET-CLIP/ret_clip_vit_b_16.pt"
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VLM checkpoints/weights")
    parser.add_argument("--hf_models", nargs="+", choices=list(VLM_HF_MODELS.keys()) + ["all", "none"], default=["all"], help="HF Models to download")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    hf_to_download = list(VLM_HF_MODELS.keys()) if "all" in args.hf_models else ([] if "none" in args.hf_models else args.hf_models)
        
    if hf_to_download:
        print(f"Downloading VLM pre-trained weights from HuggingFace: {hf_to_download}")
        for name in hf_to_download:
            repo_id = VLM_HF_MODELS[name]
            print(f"Downloading {repo_id}...")
            try:
                snapshot_download(repo_id=repo_id, token=token, repo_type="model")
                print(f"Successfully downloaded {repo_id}")
            except Exception as e:
                print(f"Failed to download {repo_id}. Error: {e}", file=sys.stderr)
                
    print("Downloading weights from Google Drive...")
    try:
        import gdown
        for name, info in VLM_GDRIVE_MODELS.items():
            if not os.path.exists(info['path']):
                print(f"Downloading {name} weights to {info['path']}...")
                gdown.download(id=info['id'], output=info['path'], quiet=False)
            else:
                print(f"{name} weights already exist at {info['path']}")
    except ImportError:
        print("gdown not installed. Skipping Google Drive downloads. Please pip install gdown.", file=sys.stderr)
        
    print("Caching TorchVision ResNet50 for FLAIR offline mode...")
    try:
        import torchvision
        import torch
        # Triggers download to ~/.cache/torch/hub/checkpoints/
        torchvision.models.resnet50(pretrained=True)
        print("Successfully cached resnet50.")
    except Exception as e:
        print(f"Failed caching resnet50. Error: {e}")

    print("Caching OpenAI CLIP ViT-B/32 for EyeCLIP offline mode...")
    try:
        import clip
        clip.load("ViT-B/32", device="cpu")
        print("Successfully cached ViT-B/32.")
    except Exception as e:
        print(f"Failed caching ViT-B/32. Error: {e}")
        
    print("Done downloading VLMs.")
