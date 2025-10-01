import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights, ViT_L_16_Weights
import torchvision.transforms as T
from agents.resnet_models import resnet_conv3_compressed, resnet_conv4_compressed, resnet_conv5, resnet18_conv5
from agents.moco_models import moco_conv5, moco_conv3_compressed, moco_conv4_compressed
from agents.vit_models import vit_s16, vit_b16, vit_l16
import mvp
from r3m import load_r3m
import re
import os
from utils import *

class ToTensorIfNot(T.ToTensor):
    def __call__(self, pic):
        if not torch.is_tensor(pic):
            return super().__call__(pic)
        return pic
    
class FeatureExtractorFactory:
    """Factory class for creating different types of feature extractors"""
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        if height!=224 or width!=224:
            ColorPrint.yellow("ATTENTION low resolution")
        assert height==width, "Currently only square inputs are supported" # if you want to change, remember to carefully change T.CenterCrop(self.height)

    
    def create_feature_extractor(self, pretrained_path):
        """Main dispatcher for creating feature extractors based on pretrained_path"""
        print(f"Creating feature extractor with path: {pretrained_path}")
        
        # Handle None or 'none' case
        if pretrained_path is None or pretrained_path.lower() == 'none':
            return self._create_vanilla_resnet18()
        
        print(os.path.exists(pretrained_path), "Path model exists or not?")
        if self._is_vit_local_checkpoint(pretrained_path):
            return self._create_vit_from_local(pretrained_path)
        elif self._is_mcr_local_checkpoint(pretrained_path):
            return self._create_mcr_from_local(pretrained_path)
        elif self._is_taco_checkpoint(pretrained_path):
            return self._create_from_taco_checkpoint(pretrained_path)
        elif os.path.exists(pretrained_path):
            return self._create_from_checkpoint(pretrained_path)
        
        # Check for special model types
        if 'mvp' in pretrained_path.lower():
            return self._create_mvp_model(pretrained_path)
        elif 'r3m' in pretrained_path.lower():
            return self._create_r3m_model(pretrained_path)
        elif self._is_vit_config_format(pretrained_path):
            return self._create_vit_from_config(pretrained_path)
        elif self._is_resnet_config_format(pretrained_path):
            return self._create_from_config(pretrained_path)
        else:
            raise ValueError(f"Unrecognized pretrained_path format: {pretrained_path}")
    
    def _create_vanilla_resnet18(self):
        """Create ResNet18 without pretrained weights"""
        print("Creating vanilla ResNet18 without pretrained weights")
        feature_extractor = models.resnet18(weights=None)
        feature_extractor = self._adapt_resnet_for_input_size(feature_extractor)
        feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove fc layer
        preprocess = self._get_imagenet_transform()
        return feature_extractor, preprocess
    
    def _create_vanilla_resnet50(self):
        """Create ResNet50 without pretrained weights"""
        print("Creating vanilla ResNet50 without pretrained weights")
        feature_extractor = models.resnet50(weights=None)
        # feature_extractor = self._adapt_resnet_for_input_size(feature_extractor)
        feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove fc layer
        preprocess = self._get_imagenet_transform()
        return feature_extractor, preprocess
    
    def _create_from_checkpoint(self, checkpoint_path):
        """Create model from checkpoint file"""
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        if 'moco' in checkpoint_path:
            return self._create_moco_model(checkpoint_path)
        elif 'resnet' in checkpoint_path:
            return self._create_resnet_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    
    def _create_moco_model(self, checkpoint_path):
        """Create MoCo model from checkpoint"""
        print(f"Loading MoCo model from {checkpoint_path}")
        
        if 'l3' in checkpoint_path:
            feature_extractor = moco_conv3_compressed(checkpoint_path)
        elif 'l4' in checkpoint_path:
            feature_extractor = moco_conv4_compressed(checkpoint_path)
        else:
            feature_extractor = moco_conv5(checkpoint_path)
        
        preprocess = self._get_imagenet_transform()
        print("MoCo model loaded successfully")
        return feature_extractor, preprocess
    
    def _create_resnet_checkpoint(self, checkpoint_path):
        """Create ResNet model from custom checkpoint"""
        print(f"Loading ResNet model from checkpoint: {checkpoint_path}")
        
        if 'resnet50_l3' in checkpoint_path:
            feature_extractor = resnet_conv3_compressed(checkpoint_path)
        elif 'resnet50_l4' in checkpoint_path:
            feature_extractor = resnet_conv4_compressed(checkpoint_path)
        elif 'resnet50_l5' in checkpoint_path:
            feature_extractor = resnet_conv5(checkpoint_path)
        elif 'resnet18_l5' in checkpoint_path:
            feature_extractor = resnet18_conv5(checkpoint_path)
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
        
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.height),
            ToTensorIfNot(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        print(f"ResNet model loaded successfully")
        return feature_extractor, preprocess
    
    def _create_mvp_model(self, pretrained_path):
        """Create MVP model"""
        print("Loading MVP model")
        raise NotImplementedError("Model not anymore available")
        feature_extractor = mvp.load("vits-mae-hoi")
        # preprocess = TO IMPLEMENT
        return feature_extractor, preprocess
    
    def _create_r3m_model(self, pretrained_path):
        """Create R3M model"""
        print("Loading R3M model")
        feature_extractor = load_r3m("resnet50")
        feature_extractor.fc = nn.Identity()  # Remove final classification layer
        preprocess = T.Compose(
        [
            ToTensorIfNot(),  # this divides by 255
            T.Resize(256),
            T.CenterCrop(self.height),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return feature_extractor, preprocess
    
    def _create_from_config(self, config_string):
        """Create ResNet from configuration string format: resnet<k>_l<n>_<initialization>"""
        print(f"Creating ResNet from config: {config_string}")
        
        match = re.match(r'resnet(\d+)_l(\d+)_(\w+)', config_string)
        if not match:
            raise ValueError(f"Invalid config format: {config_string}. Expected: resnet<k>_l<n>_<initialization>")
        
        k, n, initialization = match.groups()
        k, n = int(k), int(n)
        
        # Create base ResNet
        feature_extractor = self._create_base_resnet(k, initialization)
        
        preprocess = self._get_imagenet_transform()

        # Apply layer cutting
        feature_extractor = self._cut_resnet_at_layer(feature_extractor, n)
        
        print(f"ResNet{k} created with layer cut at l{n} and {initialization} initialization")
        return feature_extractor, preprocess
    
    def _create_base_resnet(self, k, initialization):
        """Create base ResNet architecture"""
        if k == 18:
            if initialization == 'pretrained':
                feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                feature_extractor = models.resnet18(weights=None)
                feature_extractor = self._adapt_resnet_for_input_size(feature_extractor)
        elif k == 50:
            if initialization == 'pretrained':
                feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                feature_extractor = models.resnet50(weights=None)
                feature_extractor = self._adapt_resnet_for_input_size(feature_extractor)
        else:
            raise ValueError(f"Unsupported ResNet variant: ResNet{k}")
        
        return feature_extractor
    
    def _get_imagenet_transform(self):
        """Get standard ImageNet normalization and resize transform"""

        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.height),
            ToTensorIfNot(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return preprocess
    
    def _is_resnet_config_format(self, pretrained_path):
        """Check if string matches resnet config format or is a resnet checkpoint file"""
        # Check for standard config format: resnet18_l5_pretrained
        if bool(re.match(r'resnet\d+_l\d+_\w+', pretrained_path)):
            return True
        
        # Check for file paths containing resnet (e.g., resnet18_l5.pt, resnet50_l5.tar)
        if os.path.isfile(pretrained_path) and 'resnet' in os.path.basename(pretrained_path).lower():
            return True
            
        return False
    
    def _is_vit_config_format(self, pretrained_path):
        """Check if string matches ViT config format: vit_s_scratch, vit_b_scratch, vit_l_scratch"""
        return bool(re.match(r'vit_[sbl]_\w+', pretrained_path.lower()))
    
    def _is_vit_local_checkpoint(self, pretrained_path):
        """Check if path is a local ViT checkpoint file"""
        return (pretrained_path.lower().endswith('.pt') or pretrained_path.lower().endswith('.pth'))  and ('vit_' in pretrained_path.lower())

    def _create_vit_from_config(self, config_string):
        """Create Vision Transformer from config string: vit_s_scratch, vit_b_scratch, vit_l_scratch"""
        print(f"Creating ViT from config: {config_string}")
        match = re.match(r'vit_([sbl])_(\w+)', config_string.lower())
        if not match:
            raise ValueError(f"Invalid ViT config format: {config_string}. Expected: vit_<s|b|l>_<initialization>")
        
        size, initialization = match.groups()
        
        # Use custom ViT implementations from vit_models.py
        if size == 's':
            model, hidden_dim = vit_s16("none")
        elif size == 'b':
            model, hidden_dim = vit_b16("none")
        elif size == 'l':
            model, hidden_dim = vit_l16("none")
        else:
            raise ValueError(f"Unsupported ViT size: {size}. Supported: s, b, l")
        
        # Create feature extractor wrapper
        feature_extractor = self._create_vit_feature_wrapper(model)
        preprocess = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.height),
            ToTensorIfNot(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
        
        print(f"ViT-{size.upper()} created with {initialization} initialization (hidden_dim={hidden_dim})")
        return feature_extractor, preprocess

    def _create_vit_from_local(self, checkpoint_path):
        """Load ViT model from local checkpoint"""
        print(f"Loading ViT model from local checkpoint: {checkpoint_path}")
        
        # Determine model size from path
        if 'vit_s' in checkpoint_path.lower():
            model, hidden_dim = vit_s16("none")
        elif 'vit_b' in checkpoint_path.lower():
            model, hidden_dim = vit_b16("none")
        elif 'vit_l' in checkpoint_path.lower():
            model, hidden_dim = vit_l16("none")
        else:
            raise ValueError(f"Unknown ViT variant in checkpoint path: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'encoder' in checkpoint:
            # If checkpoint contains full agent state, extract encoder
            state_dict = checkpoint['encoder']
        else:
            # Direct model state dict
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        
        feature_extractor = self._create_vit_feature_wrapper(model)
        preprocess = self._get_imagenet_transform()
        
        print(f"Loaded ViT from local checkpoint (hidden_dim={hidden_dim})")
        return feature_extractor, preprocess

    def _create_vit_feature_wrapper(self, vit_model):
        """Create wrapper for ViT model to extract features"""
        class ViTFeatureExtractor(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model
            
            def forward(self, x):
                # Use the forward method from VisionTransformer which applies norm
                return self.vit(x)
        
        return ViTFeatureExtractor(vit_model)

    def _adapt_vit_for_features(self, vit_model):
        """Adapt ViT model to output features instead of classifications (legacy method)"""
        class ViTFeatureExtractor(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model
                self.vit.heads = nn.Identity()
            def forward(self, x):
                x = self.vit(x)
                return x  # [batch_size, hidden_dim]
        return ViTFeatureExtractor(vit_model)

    def _cut_resnet_at_layer(self, feature_extractor, layer_num):
        """Cut ResNet at specified layer"""
        children = list(feature_extractor.children())
        
        if layer_num == 5:
            # Remove only fc layer
            return nn.Sequential(*children[:-1])
        elif layer_num == 4:
            # Remove fc and avgpool
            return nn.Sequential(*children[:-2])
        elif layer_num == 3:
            # Remove fc, avgpool, and layer4
            return nn.Sequential(*children[:-3])
        else:
            raise ValueError(f"Unsupported layer cut: l{layer_num}")

    def _adapt_resnet_for_input_size(self, feature_extractor):
        """Modify ResNet's first layer for non-224x224 input sizes"""
        if self.height != 224 or self.width != 224:
            kernel_size = min(7, self.height // 4, self.width // 4)
            stride = max(1, min(2, self.height // 112, self.width // 112))
            
            feature_extractor.conv1 = nn.Conv2d(
                3, 64, kernel_size=kernel_size, stride=stride, 
                padding=kernel_size//2, bias=False
            )
        return feature_extractor

    def _is_mcr_local_checkpoint(self, pretrained_path):
        """Check if path is a local MCR checkpoint file"""
        return (pretrained_path.lower().endswith('.pt') or pretrained_path.lower().endswith('.pth')) and ('mcr' in pretrained_path.lower())

    def _is_taco_checkpoint(self, pretrained_path):
        """Check if path is a TACO checkpoint file containing complete model"""
        if not (pretrained_path.lower().endswith('.pt') or pretrained_path.lower().endswith('.pth')):
            return False
        
        # Check filename for TACO indicators
        if 'taco_' in pretrained_path.lower():
            return True
            
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            # Check if it has both encoder and taco components (indicating a full TACO checkpoint)
            has_encoder = 'encoder' in checkpoint
            has_taco = 'taco' in checkpoint
            
            # Also check if encoder contains feature_extractor keys
            if has_encoder:
                encoder_keys = list(checkpoint['encoder'].keys())
                has_feature_extractor = any(key.startswith('feature_extractor.') for key in encoder_keys)
                return has_feature_extractor
                
            return has_encoder and has_taco
        except:
            return False

    def _create_from_taco_checkpoint(self, checkpoint_path):
        """Create feature extractor from a full TACO checkpoint"""
        print(f"Loading feature extractor from TACO checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'encoder' not in checkpoint:
            raise ValueError(f"No encoder found in TACO checkpoint: {checkpoint_path}")
        
        encoder_state = checkpoint['encoder']
        
        # Extract only feature_extractor keys
        feature_extractor_state = {}
        for key, value in encoder_state.items():
            if key.startswith('feature_extractor.'):
                # Remove the 'feature_extractor.' prefix
                new_key = key[len('feature_extractor.'):]
                feature_extractor_state[new_key] = value
        
        if not feature_extractor_state:
            raise ValueError(f"No feature_extractor keys found in TACO checkpoint: {checkpoint_path}")
        
        # Determine model type from checkpoint path or keys
        if 'resnet18' in checkpoint_path.lower():
            feature_extractor = models.resnet18(weights=None)
            feature_extractor = self._adapt_resnet_for_input_size(feature_extractor)
            feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove fc layer
        elif 'resnet50' in checkpoint_path.lower() or any('layer4' in key for key in feature_extractor_state.keys()):
            feature_extractor = models.resnet50(weights=None)
            feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove fc layer
        else:
            # Default to ResNet18
            ColorPrint.yellow("Could not determine ResNet variant from checkpoint, defaulting to ResNet18")
            feature_extractor = models.resnet18(weights=None)
            feature_extractor = self._adapt_resnet_for_input_size(feature_extractor)
            feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove fc layer
        
        # Load the feature extractor weights
        msg = feature_extractor.load_state_dict(feature_extractor_state, strict=False)
        if msg.missing_keys:
            ColorPrint.yellow(f"Missing keys when loading feature extractor: {msg.missing_keys}")
        if msg.unexpected_keys:
            ColorPrint.yellow(f"Unexpected keys when loading feature extractor: {msg.unexpected_keys}")
        
        preprocess = self._get_imagenet_transform()
        
        print(f"Feature extractor loaded successfully from TACO checkpoint")
        return feature_extractor, preprocess

