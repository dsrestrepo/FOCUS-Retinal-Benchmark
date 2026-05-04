import sys, os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from transformers import (
    AutoProcessor, 
    AutoModel, 
    CLIPModel, 
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor
)

class BaseVLM(ABC):
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.offline_mode = True
        if self.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
        self.model = None
        self.processor = None
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        pass
        
    @abstractmethod
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        pass

class HFCLIPModel(BaseVLM):
    def load_model(self):
        token = os.environ.get("HF_TOKEN")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, token=token, local_files_only=True)
        self.model = CLIPModel.from_pretrained(self.model_id, token=token, local_files_only=True).to(self.device).eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            img_embeds = getattr(outputs, 'image_embeds', getattr(outputs, 'pooler_output', outputs[0] if isinstance(outputs, tuple) else outputs))
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_embeds = getattr(outputs, 'text_embeds', getattr(outputs, 'pooler_output', outputs[0] if isinstance(outputs, tuple) else outputs))
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

class HFSiglipModel(BaseVLM):
    def load_model(self):
        # MedSigLIP and SigLIP2
        token = os.environ.get("HF_TOKEN")
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=token, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_id, token=token, local_files_only=True).to(self.device).eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            img_embeds = getattr(outputs, 'image_embeds', getattr(outputs, 'pooler_output', outputs[0] if isinstance(outputs, tuple) else outputs))
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_embeds = getattr(outputs, 'text_embeds', getattr(outputs, 'pooler_output', outputs[0] if isinstance(outputs, tuple) else outputs))
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

class HFFLAIR(BaseVLM):
    def load_model(self):
        # Requires: pip install git+https://github.com/jusiro/FLAIR.git
        from flair import FLAIRModel
        token = os.environ.get("HF_TOKEN")
        self.model = FLAIRModel.from_pretrained("jusiro2/FLAIR", token=token, local_files_only=True).to(self.device).eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        img_embeds_list = []
        for img in images:
            img_np = np.array(img.convert('RGB'))
            img_tensor = self.model.preprocess_image(img_np)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            with torch.no_grad():
                img_features = self.model.vision_model(img_tensor)
            img_embeds_list.append(img_features)
        
        # normalize
        embeds = torch.cat(img_embeds_list, dim=0)
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        text_input_ids, text_attention_mask = self.model.preprocess_text(texts)
        with torch.no_grad():
            text_embeds = self.model.text_model(text_input_ids.to(self.device), text_attention_mask.to(self.device))
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


class EyeCLIPModel(BaseVLM):
    def load_model(self):
        sys.path.append(os.path.abspath('ext_repos/EyeCLIP'))
        try:
            import eyeclip
        except ImportError:
            raise ImportError("EyeCLIP environment not found. Ensure ext_repos/EyeCLIP exists and its dependencies are installed.")
        
        self.model, self.processor = eyeclip.load("ViT-B/32", device=self.device, jit=False)
        checkpoint_path = "ext_repos/EyeCLIP/eyeclip_visual.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running with base clips.")
        self.model.eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        processed_images = [self.processor(img.convert("RGB")).unsqueeze(0) for img in images]
        image_tensor = torch.cat(processed_images, dim=0).to(self.device)
        with torch.no_grad():
            img_embeds = self.model.encode_image(image_tensor)
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        import eyeclip
        text_tokens = eyeclip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.encode_text(text_tokens)
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


class RETCLIPModel(BaseVLM):
    def load_model(self):
        sys.path.append(os.path.abspath('ext_repos/RET-CLIP'))
        try:
            from RET_CLIP.clip.utils import load_from_name, tokenize, image_transform
        except ImportError:
            raise ImportError("RET-CLIP environment not found. Ensure ext_repos/RET-CLIP exists.")
            
        checkpoint_path = "ext_repos/RET-CLIP/ret_clip_vit_b_16.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Expected under ext_repos/RET-CLIP.")
            checkpoint_path = "ViT-B-16"
            
        self.model, self.processor = load_from_name(
            checkpoint_path, 
            device=self.device, 
            vision_model_name="ViT-B-16", 
            text_model_name="RoBERTa-wwm-ext-base-chinese", 
            input_resolution=224
        )
        self.model.eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        processed_images = [self.processor(img.convert("RGB")).unsqueeze(0) for img in images]
        image_tensor = torch.cat(processed_images, dim=0).to(self.device)
        with torch.no_grad():
            img_embeds = self.model.encode_image(image_tensor, None)
            if isinstance(img_embeds, tuple):
                img_embeds = img_embeds[0]
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        from RET_CLIP.clip.utils import tokenize
        text_tokens = tokenize(texts).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.encode_text(text_tokens)
            if isinstance(text_embeds, tuple):
                text_embeds = text_embeds[0]
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


def get_vlm_model(model_name: str, device: str = "cuda") -> BaseVLM:
    model_name_lower = model_name.lower()
    if "flair" in model_name_lower:
        return HFFLAIR("jusiro2/FLAIR", device)
    if "eyeclip" in model_name_lower:
        return EyeCLIPModel(model_name, device)
    if "ret-clip" in model_name_lower:
        return RETCLIPModel(model_name, device)
        
    if "clip" in model_name_lower and "/" in model_name: 
        return HFCLIPModel(model_name, device)
    if "siglip" in model_name_lower:
        return HFSiglipModel(model_name, device)
    
    raise ValueError(f"Unknown VLM model: {model_name}")
