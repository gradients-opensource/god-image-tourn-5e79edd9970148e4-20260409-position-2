#!/usr/bin/env python3
"""
Caption Enhancement System
Improve caption quality for better text-image alignment
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


class CaptionEnhancer:
    """Enhance existing captions with AI-generated descriptions"""
    
    def __init__(self, model_name: str = "blip2-opt-2.7b"):
        """
        Initialize caption enhancer
        
        Args:
            model_name: Model to use for caption generation
                - blip2-opt-2.7b: Balanced quality/speed
                - blip2-flan-t5-xl: Higher quality, slower
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """Lazy load model to save memory"""
        if self.model is not None:
            return
        
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            from PIL import Image
            
            print(f"Loading {self.model_name}...")
            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{self.model_name}")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                f"Salesforce/{self.model_name}",
                device_map="auto",
                load_in_8bit=True  # Use 8-bit quantization to save memory
            )
            print("✅ Model loaded successfully")
            
        except ImportError:
            print("❌ Error: transformers library not installed")
            print("   Install with: pip install transformers accelerate bitsandbytes")
            raise
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate caption for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Generated caption
        """
        from PIL import Image
        
        self._load_model()
        
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(image, return_tensors="pt").to(
            self.model.device,
            self.model.dtype
        )
        
        generated_ids = self.model.generate(
            **inputs,
            max_length=75,
            num_beams=5,
            temperature=1.0
        )
        
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()
    
    def enhance_caption(
        self,
        original_caption: str,
        generated_caption: str,
        mode: str = "append"
    ) -> str:
        """
        Combine original and generated captions
        
        Args:
            original_caption: Existing caption
            generated_caption: AI-generated caption
            mode: How to combine ('append', 'prepend', 'replace')
            
        Returns:
            Enhanced caption
        """
        if mode == "replace":
            return generated_caption
        elif mode == "prepend":
            return f"{generated_caption}, {original_caption}"
        else:  # append
            return f"{original_caption}, {generated_caption}"
    
    def enhance_directory(
        self,
        image_dir: str,
        mode: str = "append",
        skip_existing: bool = True,
        dry_run: bool = False
    ) -> dict:
        """
        Enhance all captions in a directory
        
        Args:
            image_dir: Directory containing images and .txt captions
            mode: How to combine captions
            skip_existing: Skip if caption already looks enhanced
            dry_run: Don't actually modify files, just show what would be done
            
        Returns:
            Statistics about the enhancement process
        """
        image_dir = Path(image_dir)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
        
        if not image_files:
            print(f"⚠️  No images found in {image_dir}")
            return {}
        
        stats = {
            "total_images": len(image_files),
            "enhanced": 0,
            "skipped": 0,
            "errors": 0
        }
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Enhancing captions"):
            try:
                caption_path = image_path.with_suffix('.txt')
                
                # Read original caption
                if caption_path.exists():
                    original_caption = caption_path.read_text().strip()
                    
                    # Skip if already enhanced (heuristic: contains commas)
                    if skip_existing and original_caption.count(',') >= 3:
                        stats["skipped"] += 1
                        continue
                else:
                    original_caption = ""
                
                # Generate caption
                generated_caption = self.generate_caption(str(image_path))
                
                # Enhance
                enhanced_caption = self.enhance_caption(
                    original_caption,
                    generated_caption,
                    mode
                )
                
                # Save
                if not dry_run:
                    caption_path.write_text(enhanced_caption)
                else:
                    print(f"\n{image_path.name}:")
                    print(f"  Original: {original_caption[:100]}")
                    print(f"  Enhanced: {enhanced_caption[:100]}")
                
                stats["enhanced"] += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {image_path.name}: {e}")
                stats["errors"] += 1
        
        return stats
    
    def analyze_captions(self, image_dir: str) -> dict:
        """
        Analyze caption quality in a directory
        
        Args:
            image_dir: Directory to analyze
            
        Returns:
            Analysis statistics
        """
        image_dir = Path(image_dir)
        
        caption_files = list(image_dir.glob("*.txt"))
        
        if not caption_files:
            return {"error": "No caption files found"}
        
        total_length = 0
        total_words = 0
        total_commas = 0
        empty_captions = 0
        
        for caption_file in caption_files:
            caption = caption_file.read_text().strip()
            
            if not caption:
                empty_captions += 1
                continue
            
            total_length += len(caption)
            total_words += len(caption.split())
            total_commas += caption.count(',')
        
        num_captions = len(caption_files)
        
        return {
            "total_captions": num_captions,
            "empty_captions": empty_captions,
            "avg_length": total_length / num_captions if num_captions > 0 else 0,
            "avg_words": total_words / num_captions if num_captions > 0 else 0,
            "avg_details": total_commas / num_captions if num_captions > 0 else 0,
            "quality_score": min(100, (total_words / num_captions) * 2) if num_captions > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(
        description="Enhance image captions for better training"
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images and captions"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["append", "prepend", "replace"],
        default="append",
        help="How to combine original and generated captions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="blip2-opt-2.7b",
        choices=["blip2-opt-2.7b", "blip2-flan-t5-xl"],
        help="Model to use for caption generation"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Enhance all captions, even if they look already enhanced"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing captions without enhancing"
    )
    
    args = parser.parse_args()
    
    enhancer = CaptionEnhancer(model_name=args.model)
    
    if args.analyze_only:
        print("\n" + "=" * 80)
        print("Caption Analysis")
        print("=" * 80)
        
        stats = enhancer.analyze_captions(args.image_dir)
        
        print(f"\nDirectory: {args.image_dir}")
        print(f"Total captions: {stats.get('total_captions', 0)}")
        print(f"Empty captions: {stats.get('empty_captions', 0)}")
        print(f"Average length: {stats.get('avg_length', 0):.1f} characters")
        print(f"Average words: {stats.get('avg_words', 0):.1f}")
        print(f"Average detail level: {stats.get('avg_details', 0):.1f} descriptors")
        print(f"Quality score: {stats.get('quality_score', 0):.1f}/100")
        
        if stats.get('quality_score', 0) < 30:
            print("\n⚠️  Low quality captions detected. Enhancement recommended!")
        elif stats.get('quality_score', 0) < 60:
            print("\n✅ Moderate quality captions. Enhancement may help.")
        else:
            print("\n✅ Good quality captions!")
    
    else:
        print("\n" + "=" * 80)
        print("Caption Enhancement")
        print("=" * 80)
        print(f"Directory: {args.image_dir}")
        print(f"Mode: {args.mode}")
        print(f"Model: {args.model}")
        print(f"Dry run: {args.dry_run}")
        
        stats = enhancer.enhance_directory(
            args.image_dir,
            mode=args.mode,
            skip_existing=not args.no_skip_existing,
            dry_run=args.dry_run
        )
        
        print("\n" + "=" * 80)
        print("Enhancement Complete")
        print("=" * 80)
        print(f"Total images: {stats['total_images']}")
        print(f"Enhanced: {stats['enhanced']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Errors: {stats['errors']}")
        
        if args.dry_run:
            print("\n⚠️  This was a dry run. No files were modified.")
            print("   Remove --dry-run to apply changes.")


if __name__ == "__main__":
    main()

