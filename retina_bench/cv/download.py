import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import subprocess
from huggingface_hub import snapshot_download, hf_hub_download
from retina_bench.core.download_utils import clone_repo

CV_REPOS = {
    "RETFound": "https://github.com/rmaphoh/RETFound",
    "EyeFM": "https://github.com/eyefm/EyeFM",
    "VisionFM": "https://github.com/ABILab-CUHK/VisionFM"
}

CV_HF_MODELS = [
    "YukunZhou/RETFound_mae_natureCFP",
    "YukunZhou/RETFound_mae_natureOCT",
    "YukunZhou/RETFound_mae_meh",
    "YukunZhou/RETFound_mae_shanghai",
    "YukunZhou/RETFound_dinov2_meh",
    "YukunZhou/RETFound_dinov2_shanghai",
    "google/vit-large-patch16-224",
    "facebook/dinov2-large",
    "facebook/dinov3-vitl16-pretrain-lvd1689m"
]

VISIONFM_WEIGHTS = {
    "VFM_Fundus_weights.pth": "13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4"
}

def download_gdrive(file_id, output_path):
    print(f"Downloading {output_path} from Google Drive...")
    try:
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)
    except ImportError:
        print("gdown not installed. Falling back to subprocess...")
        subprocess.run(["gdown", file_id, "-O", output_path])
    except Exception as e:
        print(f"Failed to download {output_path}: {e}")

def download_visionfm_weights():
    # VisionFM weights are on Google Drive. 
    out_dir = os.path.join("ext_repos", "VisionFM", "pretrain_weights")
    os.makedirs(out_dir, exist_ok=True)
    
    for filename, file_id in VISIONFM_WEIGHTS.items():
        out_path = os.path.join(out_dir, filename)
        if not os.path.exists(out_path):
            download_gdrive(file_id, out_path)
        else:
            print(f"Already downloaded: {out_path}")

def download_eyefm_weights():
    # EyeFM weights from HuggingFace
    print("Downloading EyeFM weights if available on HF...")
    try:
        snapshot_download(repo_id="eyefm/eyefm-base", force_download=False)
    except Exception as e:
        print(f"Could not download EyeFM from HF. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CV models/repos/weights")
    parser.add_argument("--models", nargs="+", choices=list(CV_REPOS.keys()) + ["all"], default=["all"], help="Models to download")
    parser.add_argument("--weights-only", action="store_true", help="Download only weights")
    args = parser.parse_args()

    if not args.weights_only:
        to_download = list(CV_REPOS.keys()) if "all" in args.models else args.models
        
        print(f"Downloading CV Repositories: {to_download}")
        for name in to_download:
            clone_repo(CV_REPOS[name], name)
        print("Done downloading CV repos.")
        
    print("Downloading CV model weights and HuggingFace models...")
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    for hf_model in CV_HF_MODELS:
        print(f"Downloading {hf_model}...")
        try:
            snapshot_download(repo_id=hf_model, force_download=False, token=token, repo_type="model")
        except Exception as e:
            print(f"Failed to download {hf_model}: {e}")
            
    download_visionfm_weights()
    print("Done downloading CV weights.")
