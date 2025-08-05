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
    def __call__(self, images, prompts, timesteps=None, config=None):
        raise NotImplementedError("Subclasses must implement __call__")

class BrightnessScorer(Scorer):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None, config=None):
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
    def __call__(self, images, prompts, timesteps=None, config=None):
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
    def __call__(self, images, prompts, timesteps=None, config=None):
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
    def __call__(self, images, prompts, timesteps=None, config=None):
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

class HPSScorer(Scorer):
    def __init__(self, hps_version="v2.1", dtype=torch.float32):
        """
        Initialize a scorer that uses HPSv2 to evaluate human preferences for text-to-image generation.
        
        Args:
            hps_version (str): The HPS version to use (default: "v2.1")
            dtype: Torch data type for the output
        """
        super().__init__(dtype)
        
        # Import hpsv2 here to avoid dependency issues
        try:
            import hpsv2
            self.hpsv2 = hpsv2
        except ImportError:
            raise ImportError("hpsv2 package not found. Please install it with: pip install hpsv2")
        
        self.hps_version = hps_version

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None, config=None):
        """
        Score images based on human preferences using HPSv2.
        
        Args:
            images: List of PIL Images, torch.Tensor images, or image paths
            prompts: List of text prompts or single prompt string
            timesteps: Not used by HPSv2 (kept for compatibility)
            
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
        
        # Convert torch.Tensor images to PIL Images for HPSv2
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
            elif isinstance(img, str):
                # Assume it's a file path
                pil_img = Image.open(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            pil_images.append(pil_img)
        
        # Score each image with its corresponding prompt
        scores = []
        for prompt, pil_img in zip(prompts, pil_images):
            score = self.hpsv2.score(pil_img, prompt, hps_version=self.hps_version)
            scores.append(score)
        
        # Convert to tensor and move to appropriate device
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        scores_tensor = torch.tensor(scores, dtype=self.dtype, device=device)
        
        return scores_tensor

class CountingScorer(Scorer):
    def __init__(self, dtype=torch.float32):
        """
        Initialize a scorer that counts objects in images and compares with ground truth counts.
        
        Args:
            dtype: Torch data type for the output
        """
        super().__init__(dtype)
        
        # Import required modules
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
            import supervision as sv
            self.sv = sv  # Store supervision module as instance variable
        except ImportError as e:
            raise ImportError(f"Required packages not found: {e}. Please install transformers and supervision.")
        
        # Initialize models and components
        self._initialized = False
        self.gd_processor = None
        self.bboxmaker = None
        self.sam_processor = None
        self.segmentator = None
        self.bounding_box_annotator = None
        self.label_annotator = None
        
        self.detection_logs = []
        self.reward_logs = []

    def _initialize_models(self, config):
        """Initialize models based on config."""
        if self._initialized:
            return
            
        # Set default config values
        default_config = {
            "device": 0,
            "batch_size": 10,
            "count_reward_model": "gdsam",
            "disable_debug": False,
            "log_interval": 5,
            "class_names": "airplane, camel",
            "class_gt_counts": "3, 6",
            "reward_func": "accuracy"
        }
        
        # Update with provided config
        self.config = {**default_config, **config}
        
        print(f"Counting Scorer Config: {self.config['count_reward_model']}")
        
        if self.config['count_reward_model'] == "gdsam":
            bboxmaker_id = "IDEA-Research/grounding-dino-base"
            segmenter_id = "facebook/sam-vit-base"
            
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
            self.gd_processor = AutoProcessor.from_pretrained(bboxmaker_id)
            self.bboxmaker = AutoModelForZeroShotObjectDetection.from_pretrained(bboxmaker_id).to(f"cuda:{self.config['device']}")
            self.sam_processor = AutoProcessor.from_pretrained(segmenter_id)
            self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(f"cuda:{self.config['device']}")
            
            self.bounding_box_annotator = self.sv.BoxAnnotator()
            self.label_annotator = self.sv.LabelAnnotator(text_position=self.sv.Position.CENTER)
            
            # Parse class names and ground truth counts
            self.class_gt_counts = torch.tensor([int(n.strip()) for n in self.config['class_gt_counts'].split(",")])
            self.class_names = [t.strip() for t in self.config['class_names'].split(",")]
            self.class_texts = ". ".join(self.class_names) + "."
            
            print(f"Class texts: {self.class_texts}")
            print(f"Class names: {self.class_names}")
            print(f"Class gt counts: {self.class_gt_counts}")
        else:
            raise NotImplementedError(f"Unknown reward model: {self.config['count_reward_model']}")
        
        self.disable_debug = self.config['disable_debug']
        self.log_interval = self.config['log_interval']
        self._initialized = True

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None, config=None):
        """
        Count objects in images and return rewards based on comparison with ground truth.
        
        Args:
            images: List of PIL Images or torch.Tensor images
            prompts: Not used by counting scorer (kept for compatibility)
            timesteps: Current timestep for logging (optional)
            config: Configuration dictionary with keys:
                - class_names (str): Comma-separated class names (e.g., "horses, cars, train, airplanes")
                - class_gt_counts (str): Comma-separated ground truth counts (e.g., "5, 3, 1, 5")
                - device (int): Device to run models on (default: 0)
                - batch_size (int): Batch size for processing (default: 10)
                - count_reward_model (str): Model to use for counting (default: "gdsam")
                - disable_debug (bool): Whether to disable debug logging (default: False)
                - log_interval (int): Interval for logging (default: 5)
                - reward_func (str): Reward function type (default: "diff")
            
        Returns:
            torch.Tensor: Rewards for each image based on counting accuracy
        """
        # Initialize models if not already done or if config changed
        if config is not None:
            self._initialize_models(config)
        
        # Handle single image case
        if not isinstance(images, list):
            images = [images]
        
        # Convert torch.Tensor images to PIL Images
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
        
        # Process each image
        rewards = []
        for i, pil_img in enumerate(pil_images):
            step = timesteps[i] if timesteps is not None else i
            reward = self._gdsam_reward(pil_img, step)
            rewards.append(reward)
        
        # Convert to tensor and ensure it's on CPU for numpy compatibility
        rewards_tensor = torch.tensor(rewards, dtype=self.dtype, device='cpu')
        
        return rewards_tensor

    def _gdsam_reward(self, image, step):
        """Calculate reward using Grounding DINO + SAM approach."""
        texts = self.class_texts
        names = texts[:-1].split(". ")
        
        gd_inputs = self.gd_processor(images=image, text=texts, return_tensors="pt").to(f"cuda:{self.config['device']}")
        outputs = self.bboxmaker(**gd_inputs)
        
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            gd_inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )[0]
        
        del results["text_labels"]
        
        gt_labels = {label: i for i, label in enumerate(names)}
        results["labels"] = torch.tensor([gt_labels.get(label, -1) for label in results["labels"]])
        indices = torch.argwhere(results["labels"] >= 0).reshape(-1)
        
        for key in results:
            results[key] = results[key][indices].cpu()
        
        cnt = 0
        indices = []
        unq_labels = torch.unique(results["labels"])
        for i in unq_labels:
            idx0 = torch.argwhere(results["labels"] == i).reshape(-1)
            scores = results["scores"][idx0]
            max_score = torch.max(scores).item()
            idx1 = torch.argwhere((scores > max_score * 0.6) | (scores > 0.32)).reshape(-1)
            indices.append(idx0[idx1])
            cnt += 1
        
        counts = [0 for _ in names]
        
        if cnt:
            indices = torch.cat(indices, dim=0)
            for key in results:
                results[key] = results[key][indices]
            
            boxes = [results["boxes"].numpy().tolist()]
            
            sam_inputs = self.sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(f"cuda:{self.config['device']}")
            
            outputs = self.segmentator(**sam_inputs)
            masks = self.sam_processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=sam_inputs.original_sizes,
                reshaped_input_sizes=sam_inputs.reshaped_input_sizes
            )[0]
            masks = masks.float().permute(0, 2, 3, 1).mean(dim=-1) > 0
            
            results["masks"] = masks
            results["mask_sums"] = torch.sum(masks, dim=[1, 2]).cpu()
            
            indices0 = []
            scores = torch.zeros(results["masks"][0].shape, dtype=torch.float32, device=f"cuda:{self.config['device']}")
            
            for label_i in unq_labels:
                idx0 = torch.argwhere(results["labels"] == label_i).reshape(-1)
                cuml_mask = torch.zeros_like(results["masks"][0])
                
                for i0 in sorted(idx0.numpy().tolist(), key=lambda i: results["mask_sums"][i].item()):
                    ms = results["masks"][i0]
                    msum = results["mask_sums"][i0]
                    mpos = torch.sum(torch.bitwise_and(~cuml_mask, ms)).item()
                    
                    if mpos / msum > 0.5:
                        indices0.append(i0)
                        cuml_mask |= ms
                        scores[ms] = torch.maximum(scores[ms], torch.full_like(scores[ms], results["scores"][i0].item()))
            
            indices1 = []
            for i0 in indices0:
                ms = results["masks"][i0]
                ssum = torch.sum(scores[ms] > results["scores"][i0].item()) / results["mask_sums"][i0].item()
                if ssum < 0.5:
                    indices1.append(i0)
                    counts[results["labels"][i0].item()] += 1
            
            indices1 = torch.tensor(indices1, dtype=torch.int64)
            for key in results:
                results[key] = results[key][indices1]
        
        # Calculate reward based on difference from ground truth
        if self.config['reward_func'] == "diff":
            diff = torch.sum((self.class_gt_counts - torch.tensor(counts)) ** 2)
            reward = -diff
        elif self.config['reward_func'] == "normalized":
            # Normalized reward: 1.0 for perfect match, 0.0 for maximum deviation
            # Uses exponential decay based on relative error
            total_gt = torch.sum(self.class_gt_counts).float()
            if total_gt > 0:
                # Calculate relative error (normalized by total ground truth count)
                relative_error = torch.sum(torch.abs(self.class_gt_counts - torch.tensor(counts))) / total_gt
                # Apply exponential decay: reward = exp(-relative_error)
                # This gives 1.0 for perfect match, ~0.37 for 100% error, ~0.14 for 200% error
                reward = torch.exp(-relative_error)
            else:
                # Edge case: no ground truth objects expected
                reward = 1.0 if sum(counts) == 0 else 0.0
        elif self.config['reward_func'] == "iou":
            # IoU-like reward: intersection over union of counts
            # reward = min(gt, detected) / max(gt, detected) for each class, then average
            total_gt = torch.sum(self.class_gt_counts).float()
            total_detected = sum(counts)
            
            if total_gt > 0 and total_detected > 0:
                # Calculate IoU for each class
                iou_scores = []
                for gt_count, detected_count in zip(self.class_gt_counts, counts):
                    if gt_count > 0 or detected_count > 0:
                        intersection = min(gt_count, detected_count)
                        union = max(gt_count, detected_count)
                        iou_scores.append(intersection / union)
                    else:
                        # Both are 0, perfect match
                        iou_scores.append(1.0)
                
                # Average IoU across all classes
                reward = torch.tensor(iou_scores, dtype=self.dtype).mean()
            else:
                # Edge case: no objects expected or detected
                reward = 1.0 if total_gt == total_detected else 0.0
        elif self.config['reward_func'] == "accuracy":
            # Simple accuracy: correctly counted objects / total expected objects
            # reward = sum(min(gt, detected)) / sum(gt)
            total_gt = torch.sum(self.class_gt_counts).float()
            
            if total_gt > 0:
                # Count correctly detected objects (intersection)
                correct_count = sum(min(gt_count, detected_count) for gt_count, detected_count in zip(self.class_gt_counts, counts))
                reward = correct_count / total_gt
            else:
                # Edge case: no objects expected
                reward = 1.0 if sum(counts) == 0 else 0.0
        else:
            raise NotImplementedError(f"Unknown reward function: {self.config['reward_func']}")
        
        # Ensure reward is a CPU tensor for numpy compatibility
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu()
        else:
            reward = torch.tensor(reward, dtype=self.dtype, device='cpu')
        
        self.reward_logs.append(reward)
        
        # # Debug logging (simplified version)
        # if not self.disable_debug and (step % self.log_interval == 0):
        #     print(f"Step {step}: Counts {counts}, GT {self.class_gt_counts.tolist()}, Reward {reward}")
        
        return reward
