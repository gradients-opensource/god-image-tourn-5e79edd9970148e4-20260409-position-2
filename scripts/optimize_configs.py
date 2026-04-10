
#!/usr/bin/env python3
"""
Automated Configuration Optimizer
Apply all optimizations to existing config files
"""

import argparse
import shutil
import toml
from pathlib import Path
from typing import Dict


class ConfigOptimizer:
    """Optimize training configurations with best practices"""
    
    OPTIMIZATIONS = {
        "flux": {
            "unet_lr": 0.00003,
            "text_encoder_lr": [0.000003, 0.000003],
            "lr_scheduler": "cosine_with_restarts",
            "lr_scheduler_num_cycles": 3,
            "lr_warmup_steps": 100,
            "lr_scheduler_power": 1.5,
            "min_snr_gamma": 5,
            "gradient_accumulation_steps": 3,
            "gradient_release": True,
            "use_ema": True,
            "ema_decay": 0.9999,
            "caption_dropout_rate": 0.1,
            "caption_dropout_every_n_epochs": 3,
            "optimizer_args": [
                "weight_decay=0.008",
                "betas=(0.9,0.99)",
                "foreach=True"
            ]
        },
        "sdxl_person": {
            "optimizer_args": [
                "decouple=True",
                "d_coef=2.0",
                "weight_decay=0.0005",
                "use_bias_correction=True",
                "safeguard_warmup=True",
                "betas=(0.9,0.999)",
                "foreach=True"
            ],
            "unet_lr": 1.5,
            "text_encoder_lr": 0.8,
            "lr_warmup_steps": 50,
            "prior_loss_weight": 1.0,
            "caption_dropout_rate": 0.1,
            "caption_dropout_every_n_epochs": 3,
            "use_ema": True,
            "ema_decay": 0.9999
        },
        "sdxl_style": {
            "optimizer_type": "AdamW8bit",
            "optimizer_args": [
                "betas=(0.9, 0.999)",
                "weight_decay=5e-05",
                "eps=1e-08",
                "foreach=True"
            ],
            "unet_lr": 3e-5,
            "text_encoder_lr": 3e-6,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 50,
            "lr_scheduler_num_cycles": 2,
            "noise_offset_type": "Multires",
            "multires_noise_iterations": 8,
            "multires_noise_discount": 0.3,
            "caption_dropout_rate": 0.15,
            "caption_dropout_every_n_epochs": 2,
            "use_ema": True,
            "ema_decay": 0.9999
        }
    }
    
    @staticmethod
    def optimize_config(config_path: str, config_type: str, backup: bool = True) -> Dict:
        """
        Optimize a configuration file
        
        Args:
            config_path: Path to config file
            config_type: Type of config (flux, sdxl_person, sdxl_style)
            backup: Whether to create backup of original config
            
        Returns:
            Optimized configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Backup original
        if backup:
            backup_path = config_path.with_suffix('.toml.backup')
            shutil.copy(config_path, backup_path)
            print(f"✅ Backup created: {backup_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = toml.load(f)
        
        # Apply optimizations
        if config_type not in ConfigOptimizer.OPTIMIZATIONS:
            raise ValueError(f"Unknown config type: {config_type}")
        
        optimizations = ConfigOptimizer.OPTIMIZATIONS[config_type]
        
        # Track changes
        changes = []
        for key, value in optimizations.items():
            old_value = config.get(key)
            if old_value != value:
                config[key] = value
                changes.append(f"  {key}: {old_value} → {value}")
        
        # Save optimized config
        with open(config_path, 'w') as f:
            toml.dump(config, f)
        
        print(f"\n✅ Optimized: {config_path}")
        print(f"   Applied {len(changes)} changes:")
        for change in changes[:10]:  # Show first 10 changes
            print(change)
        if len(changes) > 10:
            print(f"   ... and {len(changes) - 10} more")
        
        return config
    
    @staticmethod
    def optimize_all_configs(config_dir: str, backup: bool = True):
        """
        Optimize all configuration files in a directory
        
        Args:
            config_dir: Directory containing config files
            backup: Whether to create backups
        """
        config_dir = Path(config_dir)
        
        configs_to_optimize = [
            ("base_diffusion_flux.toml", "flux"),
            ("base_diffusion_sdxl_person.toml", "sdxl_person"),
            ("base_diffusion_sdxl_style.toml", "sdxl_style"),
        ]
        
        print("=" * 80)
        print("Configuration Optimizer")
        print("=" * 80)
        
        for filename, config_type in configs_to_optimize:
            config_path = config_dir / filename
            if config_path.exists():
                try:
                    ConfigOptimizer.optimize_config(
                        str(config_path),
                        config_type,
                        backup=backup
                    )
                except Exception as e:
                    print(f"❌ Error optimizing {filename}: {e}")
            else:
                print(f"⚠️  Config not found: {config_path}")
        
        print("\n" + "=" * 80)
        print("Optimization Complete!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Review the optimized configs")
        print("2. Test with a small training run")
        print("3. Monitor loss curves with validation_tracker.py")
        print("4. Compare results with baseline")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize training configurations with best practices"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="scripts/core/config",
        help="Directory containing config files"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Specific config file to optimize"
    )
    parser.add_argument(
        "--config-type",
        type=str,
        choices=["flux", "sdxl_person", "sdxl_style"],
        help="Type of config (required if using --config-file)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--show-optimizations",
        action="store_true",
        help="Show available optimizations and exit"
    )
    
    args = parser.parse_args()
    
    if args.show_optimizations:
        print("\n" + "=" * 80)
        print("Available Optimizations")
        print("=" * 80)
        
        for config_type, opts in ConfigOptimizer.OPTIMIZATIONS.items():
            print(f"\n{config_type.upper()}:")
            for key, value in opts.items():
                print(f"  - {key}: {value}")
        
        return
    
    if args.config_file:
        if not args.config_type:
            parser.error("--config-type is required when using --config-file")
        
        ConfigOptimizer.optimize_config(
            args.config_file,
            args.config_type,
            backup=not args.no_backup
        )
    else:
        ConfigOptimizer.optimize_all_configs(
            args.config_dir,
            backup=not args.no_backup
        )


if __name__ == "__main__":
    main()
