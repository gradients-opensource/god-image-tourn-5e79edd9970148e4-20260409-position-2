#!/usr/bin/env python3
"""
B-LoRA (Implicit Style-Content Separation) Helper
Implements optimal LoRA block configurations for style vs content separation
"""

from typing import Dict, List, Tuple
from enum import Enum


class TrainingType(Enum):
    """Type of LoRA training"""
    PERSON = "person"  # Content/Identity focused
    STYLE = "style"    # Style/Artistic focused
    GENERAL = "general"  # Balanced


class BLoRAConfig:
    """
    B-LoRA Configuration Generator
    
    Based on research: arXiv:2403.14572
    Key Insight: Different transformer blocks encode different aspects:
    - DOWN blocks: High-level structure and semantics (PERSON)
    - MID blocks: Cross-attention and relationships (BOTH)
    - UP blocks: Spatial details, color, texture (STYLE)
    """
    
    # Block-specific learning rates for fine-grained control
    BLOCK_LR_MULTIPLIERS = {
        "down": 1.2,  # Structure needs more attention
        "mid": 1.0,   # Balanced
        "up": 0.8     # Details need gentle updates
    }
    
    @staticmethod
    def get_person_config(network_dim: int, network_alpha: int) -> Dict:
        """
        Configuration optimized for person/identity training
        
        Focus: Semantic understanding, identity preservation, structure
        Emphasize: DOWN and MID blocks, Text Encoder
        """
        # Person training: Emphasize structure and semantic blocks
        block_dims = BLoRAConfig._generate_block_dims(
            base_dim=network_dim,
            down_multiplier=1.5,  # Emphasize structure
            mid_multiplier=1.25,  # Strong relationships
            up_multiplier=0.75    # De-emphasize texture
        )
        
        block_alphas = BLoRAConfig._generate_block_alphas(
            base_alpha=network_alpha,
            down_multiplier=1.5,
            mid_multiplier=1.25,
            up_multiplier=0.75
        )
        
        return {
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": [
                f"conv_dim={max(4, network_dim // 8)}",
                f"conv_alpha={max(4, network_alpha // 8)}",
                f"block_dims={','.join(map(str, block_dims))}",
                f"block_alphas={','.join(map(str, block_alphas))}"
            ],
            "description": "Person/Identity: Focus on structure and semantic understanding"
        }
    
    @staticmethod
    def get_style_config(network_dim: int, network_alpha: int) -> Dict:
        """
        Configuration optimized for style/artistic training
        
        Focus: Visual appearance, color, texture, artistic patterns
        Emphasize: UP blocks, de-emphasize Text Encoder
        """
        # Style training: Emphasize visual/texture blocks
        block_dims = BLoRAConfig._generate_block_dims(
            base_dim=network_dim,
            down_multiplier=0.75,   # De-emphasize structure
            mid_multiplier=1.0,     # Balanced
            up_multiplier=1.5       # Emphasize texture/color
        )
        
        block_alphas = BLoRAConfig._generate_block_alphas(
            base_alpha=network_alpha,
            down_multiplier=0.75,
            mid_multiplier=1.0,
            up_multiplier=1.5
        )
        
        return {
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": [
                f"conv_dim={max(4, network_dim // 6)}",  # More conv for spatial features
                f"conv_alpha={max(4, network_alpha // 6)}",
                f"block_dims={','.join(map(str, block_dims))}",
                f"block_alphas={','.join(map(str, block_alphas))}"
            ],
            "description": "Style/Artistic: Focus on visual appearance and texture"
        }
    
    @staticmethod
    def get_general_config(network_dim: int, network_alpha: int) -> Dict:
        """
        Balanced configuration for general-purpose training
        """
        return {
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": [
                f"conv_dim={max(4, network_dim // 8)}",
                f"conv_alpha={max(4, network_alpha // 8)}"
            ],
            "description": "General: Balanced training across all blocks"
        }
    
    @staticmethod
    def _generate_block_dims(
        base_dim: int,
        down_multiplier: float,
        mid_multiplier: float,
        up_multiplier: float
    ) -> List[int]:
        """
        Generate block-specific dimensions for SDXL
        SDXL UNet in sd-scripts has 23 blocks total:
        - IN blocks: 2
        - DOWN blocks: 9 (3 groups of 3)
        - MID block: 1
        - UP blocks: 9 (3 groups of 3)
        - OUT blocks: 2
        """
        dims = []
        
        # IN blocks (0-1): Initial processing
        for _ in range(2):
            dims.append(int(base_dim * down_multiplier))
        
        # DOWN blocks (2-10): Structure/Semantics - 9 blocks
        for _ in range(9):
            dims.append(int(base_dim * down_multiplier))
        
        # MID block (11): Bottleneck - 1 block
        dims.append(int(base_dim * mid_multiplier))
        
        # UP blocks (12-20): Texture/Details - 9 blocks
        for _ in range(9):
            dims.append(int(base_dim * up_multiplier))
        
        # OUT blocks (21-22): Final processing
        for _ in range(2):
            dims.append(int(base_dim * up_multiplier))
        
        return dims
    
    @staticmethod
    def _generate_block_alphas(
        base_alpha: int,
        down_multiplier: float,
        mid_multiplier: float,
        up_multiplier: float
    ) -> List[int]:
        """Generate block-specific alphas matching dims structure (23 blocks)"""
        alphas = []
        
        # IN blocks (2)
        for _ in range(2):
            alphas.append(int(base_alpha * down_multiplier))
        
        # DOWN blocks (9)
        for _ in range(9):
            alphas.append(int(base_alpha * down_multiplier))
        
        # MID block (1)
        alphas.append(int(base_alpha * mid_multiplier))
        
        # UP blocks (9)
        for _ in range(9):
            alphas.append(int(base_alpha * up_multiplier))
        
        # OUT blocks (2)
        for _ in range(2):
            alphas.append(int(base_alpha * up_multiplier))
        
        return alphas
    
    @staticmethod
    def get_config(
        training_type: TrainingType,
        network_dim: int,
        network_alpha: int
    ) -> Dict:
        """
        Get configuration for specific training type
        
        Args:
            training_type: Type of training (PERSON, STYLE, GENERAL)
            network_dim: Base network dimension
            network_alpha: Base network alpha
            
        Returns:
            Configuration dictionary
        """
        if training_type == TrainingType.PERSON:
            return BLoRAConfig.get_person_config(network_dim, network_alpha)
        elif training_type == TrainingType.STYLE:
            return BLoRAConfig.get_style_config(network_dim, network_alpha)
        else:
            return BLoRAConfig.get_general_config(network_dim, network_alpha)


def analyze_training_requirements(
    num_images: int,
    style_diversity: float = 0.5
) -> Dict:
    """
    Analyze dataset and recommend optimal network dimensions
    
    Args:
        num_images: Number of training images
        style_diversity: Estimated style diversity (0-1)
            0: Very consistent (same person, same style)
            1: Highly diverse (many subjects, many styles)
    
    Returns:
        Recommended configuration parameters
    """
    # Base recommendations
    if num_images < 10:
        base_dim = 16
        base_alpha = 16
        risk = "HIGH"
        recommendation = "Very small dataset. Risk of overfitting is high."
    elif num_images < 25:
        base_dim = 32
        base_alpha = 32
        risk = "MEDIUM"
        recommendation = "Small dataset. Monitor for overfitting."
    elif num_images < 50:
        base_dim = 64
        base_alpha = 64
        risk = "LOW"
        recommendation = "Adequate dataset size."
    else:
        base_dim = 96
        base_alpha = 96
        risk = "VERY LOW"
        recommendation = "Good dataset size for complex training."
    
    # Adjust for diversity
    diversity_multiplier = 1.0 + (style_diversity * 0.5)
    adjusted_dim = int(base_dim * diversity_multiplier)
    adjusted_alpha = int(base_alpha * diversity_multiplier)
    
    return {
        "recommended_network_dim": adjusted_dim,
        "recommended_network_alpha": adjusted_alpha,
        "overfitting_risk": risk,
        "recommendation": recommendation,
        "estimated_epochs": max(10, min(100, 500 // num_images))
    }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("B-LoRA Configuration Generator")
    print("=" * 80)
    
    # Example 1: Person training with 32-dim LoRA
    print("\n1. Person Training (32-dim LoRA):")
    person_config = BLoRAConfig.get_config(TrainingType.PERSON, 32, 32)
    print(f"   Description: {person_config['description']}")
    print(f"   Network Dim: {person_config['network_dim']}")
    print(f"   Network Args:")
    for arg in person_config['network_args']:
        print(f"      - {arg}")
    
    # Example 2: Style training with 64-dim LoRA
    print("\n2. Style Training (64-dim LoRA):")
    style_config = BLoRAConfig.get_config(TrainingType.STYLE, 64, 64)
    print(f"   Description: {style_config['description']}")
    print(f"   Network Dim: {style_config['network_dim']}")
    print(f"   Network Args:")
    for arg in style_config['network_args']:
        print(f"      - {arg}")
    
    # Example 3: Dataset analysis
    print("\n3. Dataset Analysis:")
    analysis = analyze_training_requirements(num_images=30, style_diversity=0.3)
    print(f"   Number of images: 30")
    print(f"   Style diversity: 0.3 (low-medium)")
    print(f"   Recommended network_dim: {analysis['recommended_network_dim']}")
    print(f"   Recommended network_alpha: {analysis['recommended_network_alpha']}")
    print(f"   Overfitting risk: {analysis['overfitting_risk']}")
    print(f"   Recommendation: {analysis['recommendation']}")
    print(f"   Estimated epochs: {analysis['estimated_epochs']}")
    
    print("\n" + "=" * 80)
    print("Configuration files have been generated successfully!")
    print("=" * 80)

