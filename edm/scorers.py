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
            # Convert PIL images to tensors
            images = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images]
            images = torch.stack(images)
        
        # Convert images to float32 if they are uint8
        if images.dtype == torch.uint8:
            images = images.float() / 255.0  # Normalize to [0, 1] range
        
        # Apply perceived luminance formula (assumes RGB input: 0.2126*R + 0.7152*G + 0.0722*B)
        # Ensure images are in [C, H, W] format with C=3 for RGB
        if images.size(1) == 3:  # Channel dimension is at index 1
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=images.device).view(1, 3, 1, 1)
            luminance = (images * weights).sum(dim=1).mean(dim=(1, 2))  # Sum over channels, mean over H,W
        else:
            # Fall back to average brightness if not RGB
            luminance = images.mean(dim=(1, 2))
        
        # Ensure the luminance values are in the range [0, 1]
        # For uint8 images converted to float, they're already normalized
        # For float images that might have values outside [0, 1], we clamp them
        luminance = torch.clamp(luminance, 0.0, 1.0)
            
        return luminance

class ImageNetScorer(Scorer):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)
        
        # URL for the classifier weights
        url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
        
        # Create a directory for downloaded models if it doesn't exist
        cache_dir = os.path.expanduser("~/.cache/imagenet_classifier")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Path to save the downloaded model
        model_path = os.path.join(cache_dir, "64x64_classifier.pt")
        
        # Download the model if it doesn't exist
        if not os.path.exists(model_path):
            print(f"Downloading classifier model from {url}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")
        
        # Create model with parameters matching OpenAI's implementation
        self.model = self.create_classifier(
            image_size=64,
            classifier_use_fp16=False,
            classifier_width=128,
            classifier_depth=4,
            classifier_attention_resolutions="32,16,8",
            classifier_use_scale_shift_norm=True,
            classifier_resblock_updown=True,
            classifier_pool="attention"
        )
        
        # Load the pretrained weights
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Default normalization for ImageNet models
        self.transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def create_classifier(
        self,
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    ):
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in classifier_attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        
        from unet import EncoderUNetModel

        return EncoderUNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=classifier_width,
            out_channels=1000,
            num_res_blocks=classifier_depth,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            use_fp16=classifier_use_fp16,
            num_head_channels=64,
            use_scale_shift_norm=classifier_use_scale_shift_norm,
            resblock_updown=classifier_resblock_updown,
            pool=classifier_pool,
        )

    @torch.no_grad()
    def __call__(self, images, class_labels, timesteps):
        device = next(self.parameters()).device
        
        # Process images
        if isinstance(images, list):  # List of PIL images
            processed_images = torch.stack([self.transform(img).to(device) for img in images])
        elif isinstance(images, torch.Tensor):  # Tensor of shape [B, C, H, W]
            # If images are tensors, ensure they're in the right format
            if images.dtype == torch.uint8:
                # Convert from [0, 255] to [0, 1]
                images = images.float() / 255.0
            
            # Use images directly, assuming they're already properly formatted
            processed_images = images.to(device)
        
        # Create dummy timesteps (zeros) if not provided
        timesteps = timesteps.to(device)
        
        # Get model predictions - passing the required timesteps parameter
        logits = self.model(processed_images, timesteps)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the target class indices
        if class_labels.dim() > 1:  # One-hot encoded
            target_classes = torch.argmax(class_labels, dim=1)
        else:  # Class indices
            target_classes = class_labels
            
        # Ensure indices are on same device as probs
        scores = probs[torch.arange(probs.size(0), device=device), target_classes.to(device)]
        
        return scores

class CompressibilityScorer(Scorer):
    def __init__(self, quality=80, min_size=0, max_size=3000, dtype=torch.float32):
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