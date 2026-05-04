import sys, os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from transformers import (
    AutoImageProcessor, 
    AutoModel, 
    AutoConfig,
    ViTModel,
    ViTImageProcessor,
    Dinov2Model
)

class BaseCVModel(ABC):
    def __init__(self, model_id: str, device: str = "cuda", pooling: str = "cls"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pooling = pooling.lower()
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

    def _apply_pooling(self, img_embeds: torch.Tensor) -> torch.Tensor:
        dim_msg = f"[{self.model_id}] Original extracted feature shape: {img_embeds.shape}"
        if img_embeds.dim() == 3:
            if self.pooling == "gap":
                img_embeds = img_embeds[:, 1:].mean(dim=1)
                dim_msg += f" -> Sliced GAP (tokens 1:) to: {img_embeds.shape}"
            else:
                img_embeds = img_embeds[:, 0]
                dim_msg += f" -> Sliced [:, 0] (CLS token) to: {img_embeds.shape}"
            #print(dim_msg, flush=True)
        return img_embeds

class GeneralViTModel(BaseCVModel):
    def load_model(self):
        token = os.environ.get("HF_TOKEN")
        self.processor = ViTImageProcessor.from_pretrained(self.model_id, token=token, local_files_only=True)
        self.model = ViTModel.from_pretrained(self.model_id, token=token, local_files_only=True).to(self.device).eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # HF outputs.last_hidden_state is always 3D: [B, N, D]
            img_embeds = getattr(outputs, 'pooler_output', outputs.last_hidden_state)
            img_embeds = self._apply_pooling(img_embeds)
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

class GeneralDinoModel(BaseCVModel):
    def load_model(self):
        token = os.environ.get("HF_TOKEN")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, token=token, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_id, token=token, local_files_only=True).to(self.device).eval()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_embeds = getattr(outputs, 'pooler_output', outputs.last_hidden_state)
            img_embeds = self._apply_pooling(img_embeds)
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)


class RETFoundModel(BaseCVModel):
    def load_model(self):
        token = os.environ.get("HF_TOKEN")
        sys.path.append(os.path.abspath('ext_repos/RETFound'))
        import models_vit
        import timm
        from huggingface_hub import hf_hub_download
        
        # YukunZhou repositories might not have a preprocessor_config.json, so we fallback to foundation architectures
        model_name_only = self.model_id.split('/')[-1]
        weight_file = f"{model_name_only}.pth"
        
        try:
            checkpoint_path = hf_hub_download(repo_id=self.model_id, filename=weight_file, token=token, local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"Could not find local cached .pth weight file for {self.model_id}. Error: {e}")

        if "dinov2" in self.model_id.lower():
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", local_files_only=True)
            self.model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(self.device).eval()
        else:
            self.processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224", local_files_only=True)
            self.model = models_vit.RETFound_mae(num_classes=0).to(self.device).eval()

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found.")

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pixel_values = inputs.get('pixel_values')
            img_embeds = self.model.forward_features(pixel_values)
            img_embeds = self._apply_pooling(img_embeds)
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

class VisionFMModel(BaseCVModel):
    def load_model(self):
        visionfm_path = os.path.abspath('ext_repos/VisionFM')
        if visionfm_path not in sys.path:
            sys.path.insert(0, visionfm_path)
        try:
            # Note: VisionFM encoders logic will go here
            # Since VisionFM offers different weights per modality (Fundus, OCT, etc.),
            # One must load their specific ViT and inject weights.
            import timm
            import types
            if 'miseval' not in sys.modules:
                sys.modules['miseval'] = types.ModuleType('miseval')
                sys.modules['miseval'].evaluate = lambda *args, **kwargs: None

            from models.vision_transformer import vit_base
            import utils as visionfm_utils

            self.model = vit_base(
                img_size=[224],
                patch_size=16,
                num_classes=0, 
                use_mean_pooling=False
            ).to(self.device)
            
            # Dummy load path, requires manual override or config mapping
            checkpoint_path = "ext_repos/VisionFM/pretrain_weights/VFM_Fundus_weights.pth"
            if "oct" in self.model_id.lower():
                checkpoint_path = "ext_repos/VisionFM/pretrain_weights/VFM_OCT_weights.pth"
            
            if os.path.exists(checkpoint_path):
                # We need to allowlist numpy.core.multiarray.scalar since VisionFM's checkpoints
                # use it and PyTorch 2.6 sets weights_only=True by default for torch.load.
                import numpy
                try: # For PyTorch >= 2.6
                    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype])
                except AttributeError:
                    pass
                
                # Monkeypatch torch.load to bypass weights_only since it's hardcoded to True optionally now inside their util
                import builtins
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                try:
                    # VisionFM uses DINO-style weights. This utility extracts 'teacher' and drops prefixes like 'backbone.'
                    visionfm_utils.load_pretrained_weights(self.model, checkpoint_path, "teacher", "vit_base", 16)
                finally:
                    torch.load = original_load
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Ensure VisionFM weights are downloaded.")
            self.model.eval()
            self.processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224", local_files_only=True) # generic ViT processor
        except Exception as e:
            raise ImportError(f"VisionFM environment not found, or timm isn't installed. Ensure ext_repos/VisionFM exists. Error: {e}")
        finally:
            if visionfm_path in sys.path:
                sys.path.remove(visionfm_path)

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pixel_values = inputs.get('pixel_values')
            # VisionFM's custom VisionTransformer doesn't have forward_features, so we just use forward
            img_embeds = self.model(pixel_values, return_all_tokens=True)
            img_embeds = self._apply_pooling(img_embeds)
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

class EyeFMModel(BaseCVModel):
    def load_model(self):
        sys.path.append(os.path.abspath('ext_repos/EyeFM'))
        try:
            # Assuming EyeFM uses standard open_clip or timm architectures.
            # Loading routine specific to EyeFM
            import timm
            self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0).to(self.device)
            # You can substitute this with real EyeFM initialization
            self.model.eval()
            self.processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224", local_files_only=True)
        except ImportError:
            raise ImportError("EyeFM environment not found. Ensure ext_repos/EyeFM exists.")

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pixel_values = inputs.get('pixel_values')
            img_embeds = self.model.forward_features(pixel_values)
            img_embeds = self._apply_pooling(img_embeds)
        return img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)


def get_cv_model(model_name: str, device: str = "cuda", pooling: str = "cls") -> BaseCVModel:
    model_name_lower = model_name.lower()
    
    if "retfound" in model_name_lower:
        return RETFoundModel(model_name, device, pooling=pooling)
    if "visionfm" in model_name_lower:
        return VisionFMModel(model_name, device, pooling=pooling)
    if "eyefm" in model_name_lower:
        return EyeFMModel(model_name, device, pooling=pooling)
    
    if "dinov" in model_name_lower or "dino" in model_name_lower:
        return GeneralDinoModel(model_name, device, pooling=pooling)
    if "vit" in model_name_lower:
        return GeneralViTModel(model_name, device, pooling=pooling)
        
    raise ValueError(f"Unknown CV model: {model_name}")
