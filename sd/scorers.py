import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification, ViTImageProcessor
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import numpy as np
from PIL import Image
import os
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
import io

class Scorer(torch.nn.Module):
    """Base class for all scorers"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        raise NotImplementedError("Subclasses must implement __call__")

class BrightnessScorer(Scorer):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps):
        if isinstance(images, list):
            processed = []
            for img in images:
                # Handle torch.Tensor inputs
                if isinstance(img, torch.Tensor):
                    # Move to CPU if necessary and detach from computation graph
                    if img.device.type != "cpu":
                        img = img.detach().cpu()

                    # If the tensor already has a batch dimension (B, C, H, W) collapse it
                    if img.dim() == 4:
                        processed.append(img)  # keep as is, will be concatenated later
                    else:
                        # Assume (C, H, W)
                        processed.append(img.unsqueeze(0))
                else:
                    # Assume PIL.Image or numpy array, convert to tensor and reorder channels
                    tensor_img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
                    processed.append(tensor_img)

            # Concatenate along the batch dimension
            images = torch.cat(processed, dim=0)
        
        # Convert images to float32 if they are uint8
        if images.dtype == torch.uint8:
            images = images.float() / 255.0  # Normalize to [0, 1] range
        
        # Apply perceived luminance formula (assumes RGB input: 0.2126*R + 0.7152*G + 0.0722*B)
        # Ensure images are in [C, H, W] format with C=3 for RGB
        if images.size(1) == 3:  # Channel dimension is at index 1
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=images.device).view(1, 3, 1, 1)
            luminance = (images * weights).sum(dim=1).mean(dim=(1, 2))  # Sum over channels, mean over H,W
        else:
            # Fall back to average brightness if not standard RGB; average across all spatial dimensions
            # Supports inputs of shape [B, C, H, W] with arbitrary C as well as [B, H, W].
            if images.dim() == 4:
                luminance = images.mean(dim=(1, 2, 3))
            else:
                luminance = images.mean(dim=(1, 2))
        
        # Ensure the luminance values are in the range [0, 1]
        # For uint8 images converted to float, they're already normalized
        # For float images that might have values outside [0, 1], we clamp them
        luminance = torch.clamp(luminance, 0.0, 1.0)
            
        return luminance

class CompressibilityScorer(Scorer):
    def __init__(self, quality=80, min_size=0, max_size=150000, dtype=torch.float32):
        """
        Initialize a compressibility scorer that rewards images with higher compression rates.
        
        Args:
            quality (int): JPEG quality parameter (1-100)
            min_size (int): Expected minimum compressed size in bytes (maps to score of 1.0)
            max_size (int): Expected maximum compressed size in bytes (maps to score of 0.0)
            dtype: Torch data type for the output
        """
        super().__init__(dtype)
        self.quality = quality
        self.min_size = min_size
        self.max_size = max_size

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps):
        images = [im.cpu().squeeze(0) for im in images]
        if isinstance(images, torch.Tensor):
            # Convert tensor to numpy array
            if images.dim() == 4:  # batch of images
                return torch.tensor([self._calculate_score(img.cpu().numpy()) for img in images])
            else:  # single image
                return torch.tensor([self._calculate_score(images.cpu().numpy())])
        elif isinstance(images, list):
            # List of PIL images
            return torch.tensor([self._calculate_score(np.array(img)) for img in images])
        else:
            # Single PIL image
            return torch.tensor([self._calculate_score(np.array(images))])

    def _calculate_score(self, image):
        """
        Calculate the reward as a normalized value [0,1] based on JPEG compression size.
        Higher compressibility (smaller file size) scores closer to 1.0.
        
        Args:
            image (np.ndarray): The input image as a numpy array.
            
        Returns:
            float: The normalized reward in range [0,1]
        """
        # Handle different image shapes and formats
        if image.ndim == 3:
            if image.shape[0] == 1 or image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))  # Convert to HWC
            
            if image.shape[2] == 1:  # HW1 format (grayscale)
                image = image.squeeze(2)  # Convert to HW
                
        # Ensure image has correct shape for PIL (HW or HWC with 3 channels)
        if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in [1, 3, 4])):
            raise ValueError(f"Invalid image shape: {image.shape}")
                
        # Ensure image is in uint8 format for PIL
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        buffer = io.BytesIO()
        img = Image.fromarray(image)
        img.save(buffer, format="JPEG", quality=self.quality)
        compressed_size = len(buffer.getvalue())
        
        # Normalize to [0,1] range where 1.0 means highly compressible (small size)
        normalized_score = 1.0 - min(1.0, max(0.0, (compressed_size - self.min_size) / (self.max_size - self.min_size)))
        return normalized_score

class CLIPScorer(Scorer):
    def __init__(self, model_id="openai/clip-vit-large-patch14", dtype=torch.float32):
        """
        Initialize a scorer that measures cosine similarity between CLIP embeddings of prompts and images.
        
        Args:
            model_id (str): The HuggingFace model ID for CLIP
            dtype: Torch data type for the output
        """
        super().__init__(dtype)
        
        # Load CLIP model and processor
        self.clip = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.clip.eval()

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None):
        device = next(self.parameters()).device
        
        # Process images
        if isinstance(images, torch.Tensor) and images.dtype == torch.float32 and images.max() <= 1.0:
            do_rescale = False
        else:
            do_rescale = True
        
        # Process images with appropriate rescaling setting
        image_inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale)
        image_inputs = {k: v.to(self.dtype).to(device) for k, v in image_inputs.items()}
        
        # Get image embeddings
        image_embeds = self.clip.get_image_features(**image_inputs)
        image_embeds = image_embeds / torch.linalg.vector_norm(image_embeds, dim=-1, keepdim=True)
        
        # Handle text prompts
        if prompts is None:
            # If no prompts provided, return zeros
            return torch.zeros(image_embeds.shape[0], device=device)
        
        # Ensure prompts is a list
        if not isinstance(prompts, list):
            prompts = [prompts] * image_embeds.shape[0]
        elif len(prompts) == 1 and image_embeds.shape[0] > 1:
            prompts = prompts * image_embeds.shape[0]
        
        # Process text prompts
        text_inputs = self.processor.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(device)
        
        # Get text embeddings
        text_embeds = self.clip.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.linalg.vector_norm(text_embeds, dim=-1, keepdim=True)
        
        # Ensure consistent dtype for calculating similarity
        image_embeds = image_embeds.to(self.dtype)
        text_embeds = text_embeds.to(self.dtype)
        
        # Calculate cosine similarity
        similarities = torch.sum(image_embeds * text_embeds, dim=1)
        
        return similarities

class ImageRewardScorer(Scorer):
    def __init__(self, model_id="ImageReward-v1.0", dtype=torch.float32):
        """
        Initialize a scorer that uses ImageReward to evaluate human preferences for text-to-image generation.
        
        Args:
            model_id (str): The ImageReward model ID (default: "ImageReward-v1.0")
            dtype: Torch data type for the output
        """
        super().__init__(dtype)
        
        # Import ImageReward here to avoid dependency issues
        try:
            import ImageReward as reward
            self.reward_model = reward.load(model_id)
        except ImportError:
            raise ImportError("ImageReward package not found. Please install it with: pip install image-reward")

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None):
        """
        Score images based on human preferences using ImageReward.
        
        Args:
            images: List of PIL Images or torch.Tensor images
            prompts: List of text prompts or single prompt string
            timesteps: Not used by ImageReward (kept for compatibility)
            
        Returns:
            torch.Tensor: Human preference scores for each image
        """
        # Handle single image case
        if not isinstance(images, list):
            images = [images]
        
        # Handle single prompt case
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)
        elif len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        
        # Ensure we have the same number of prompts as images
        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
        
        # Convert torch.Tensor images to PIL Images for ImageReward
        pil_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                # Handle different tensor formats
                if img.dim() == 4:  # [B, C, H, W] - take first image
                    img = img.squeeze(0)
                elif img.dim() == 3:  # [C, H, W]
                    pass
                else:
                    raise ValueError(f"Unexpected tensor shape: {img.shape}")
                
                # Convert to numpy and transpose if needed
                img_np = img.cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HWC
                
                # Handle grayscale
                if img_np.shape[2] == 1:
                    img_np = img_np.squeeze(2)
                
                # Convert to uint8 if needed
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                
                pil_img = Image.fromarray(img_np)
            elif isinstance(img, np.ndarray):
                # Handle numpy array
                if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW format
                    img = np.transpose(img, (1, 2, 0))  # Convert to HWC
                if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                    img = img.squeeze(2)
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                pil_img = Image.fromarray(img)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            pil_images.append(pil_img)
        
        # Score each image with its corresponding prompt
        scores = []
        for prompt, pil_img in zip(prompts, pil_images):
            score = self.reward_model.score(prompt, pil_img)
            scores.append(score)
        
        # Convert to tensor and move to appropriate device
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        scores_tensor = torch.tensor(scores, dtype=self.dtype, device=device)
        
        return scores_tensor

