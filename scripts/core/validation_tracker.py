#!/usr/bin/env python3
"""
Validation Loss Tracking System
Monitors training progress and implements early stopping
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import time


class ValidationLossTracker:
    """
    Track validation loss during training for early stopping and monitoring
    """
    
    def __init__(
        self,
        output_dir: str,
        validation_every_n_steps: int = 50,
        patience: int = 5,
        min_delta: float = 0.001
    ):
        """
        Args:
            output_dir: Directory to save metrics
            validation_every_n_steps: How often to validate
            patience: Number of validations without improvement before suggesting early stop
            min_delta: Minimum change to consider as improvement
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_every_n_steps = validation_every_n_steps
        self.patience = patience
        self.min_delta = min_delta
        
        self.metrics_file = self.output_dir / "training_metrics.json"
        self.loss_history: List[Dict] = []
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0
        
        # Load existing history if available
        self._load_history()
    
    def should_validate(self, current_step: int) -> bool:
        """Check if validation should be performed at this step"""
        return current_step % self.validation_every_n_steps == 0
    
    def log_metrics(
        self, 
        step: int, 
        train_loss: float, 
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        additional_metrics: Optional[Dict] = None
    ):
        """
        Log training metrics
        
        Args:
            step: Current training step
            train_loss: Training loss value
            val_loss: Validation loss (if available)
            learning_rate: Current learning rate
            additional_metrics: Any other metrics to log
        """
        metric_entry = {
            "step": step,
            "timestamp": time.time(),
            "train_loss": train_loss,
        }
        
        if val_loss is not None:
            metric_entry["val_loss"] = val_loss
            metric_entry["overfitting_ratio"] = val_loss / train_loss if train_loss > 0 else 1.0
            
            # Check for improvement
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.steps_without_improvement = 0
                metric_entry["is_best"] = True
            else:
                self.steps_without_improvement += 1
                metric_entry["is_best"] = False
        
        if learning_rate is not None:
            metric_entry["learning_rate"] = learning_rate
        
        if additional_metrics:
            metric_entry.update(additional_metrics)
        
        self.loss_history.append(metric_entry)
        self._save_history()
        
        # Print summary
        self._print_summary(metric_entry)
    
    def check_early_stopping(self) -> bool:
        """
        Check if training should be stopped early
        
        Returns:
            True if early stopping criteria met
        """
        if len(self.loss_history) < self.patience:
            return False
        
        # Check if we haven't improved in 'patience' validations
        should_stop = self.steps_without_improvement >= self.patience
        
        if should_stop:
            print(f"\n⚠️  Early stopping triggered!")
            print(f"    No improvement for {self.patience} consecutive validations")
            print(f"    Best validation loss: {self.best_val_loss:.6f}")
        
        return should_stop
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        if not self.loss_history:
            return {}
        
        recent_losses = [entry["train_loss"] for entry in self.loss_history[-10:]]
        
        stats = {
            "total_steps": len(self.loss_history),
            "current_train_loss": recent_losses[-1] if recent_losses else None,
            "avg_recent_train_loss": sum(recent_losses) / len(recent_losses) if recent_losses else None,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float('inf') else None,
            "steps_without_improvement": self.steps_without_improvement,
        }
        
        # Calculate loss reduction rate
        if len(self.loss_history) >= 100:
            initial_loss = self.loss_history[0]["train_loss"]
            current_loss = self.loss_history[-1]["train_loss"]
            steps_elapsed = self.loss_history[-1]["step"]
            stats["loss_reduction_rate"] = (initial_loss - current_loss) / steps_elapsed * 100
        
        return stats
    
    def plot_loss_curves(self, save_path: Optional[str] = None):
        """
        Plot loss curves (requires matplotlib)
        
        Args:
            save_path: Path to save plot (if None, uses default location)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  Matplotlib not installed. Skipping plot generation.")
            return
        
        if not self.loss_history:
            print("⚠️  No history to plot")
            return
        
        steps = [entry["step"] for entry in self.loss_history]
        train_losses = [entry["train_loss"] for entry in self.loss_history]
        
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(steps, train_losses, label="Training Loss", color='blue', alpha=0.7)
        
        # Plot validation loss if available
        val_steps = [entry["step"] for entry in self.loss_history if "val_loss" in entry]
        val_losses = [entry["val_loss"] for entry in self.loss_history if "val_loss" in entry]
        if val_losses:
            plt.plot(val_steps, val_losses, label="Validation Loss", color='red', alpha=0.7, marker='o')
        
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot overfitting ratio if available
        if val_losses:
            plt.subplot(1, 2, 2)
            overfitting_ratios = [entry.get("overfitting_ratio", 1.0) for entry in self.loss_history if "val_loss" in entry]
            plt.plot(val_steps, overfitting_ratios, label="Val/Train Ratio", color='green', marker='o')
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label="Perfect fit")
            plt.axhline(y=1.15, color='orange', linestyle='--', alpha=0.5, label="15% tolerance")
            plt.xlabel("Step")
            plt.ylabel("Overfitting Ratio")
            plt.title("Overfitting Monitor")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Loss curves saved to: {save_path}")
        plt.close()
    
    def _load_history(self):
        """Load existing metrics history"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.loss_history = json.load(f)
                
                # Restore best val loss
                val_losses = [entry.get("val_loss") for entry in self.loss_history if "val_loss" in entry]
                if val_losses:
                    self.best_val_loss = min(val_losses)
                
                print(f"📂 Loaded {len(self.loss_history)} existing metric entries")
            except Exception as e:
                print(f"⚠️  Error loading history: {e}")
                self.loss_history = []
    
    def _save_history(self):
        """Save metrics history to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving metrics: {e}")
    
    def _print_summary(self, metric_entry: Dict):
        """Print formatted metric summary"""
        step = metric_entry["step"]
        train_loss = metric_entry["train_loss"]
        
        summary = f"📈 Step {step:6d} | Train Loss: {train_loss:.6f}"
        
        if "val_loss" in metric_entry:
            val_loss = metric_entry["val_loss"]
            ratio = metric_entry["overfitting_ratio"]
            is_best = "🌟" if metric_entry.get("is_best") else "  "
            summary += f" | Val Loss: {val_loss:.6f} | Ratio: {ratio:.3f} {is_best}"
        
        if "learning_rate" in metric_entry:
            lr = metric_entry["learning_rate"]
            summary += f" | LR: {lr:.2e}"
        
        print(summary)


class MetricsAggregator:
    """
    Aggregate and analyze metrics across multiple training runs
    """
    
    @staticmethod
    def compare_runs(run_dirs: List[str]) -> Dict:
        """
        Compare metrics across multiple runs
        
        Args:
            run_dirs: List of directories containing training_metrics.json
            
        Returns:
            Comparison statistics
        """
        run_stats = {}
        
        for run_dir in run_dirs:
            metrics_file = Path(run_dir) / "training_metrics.json"
            if not metrics_file.exists():
                continue
            
            with open(metrics_file, 'r') as f:
                history = json.load(f)
            
            if not history:
                continue
            
            # Extract statistics
            final_loss = history[-1].get("train_loss")
            val_losses = [entry.get("val_loss") for entry in history if "val_loss" in entry]
            best_val = min(val_losses) if val_losses else None
            
            run_stats[run_dir] = {
                "final_train_loss": final_loss,
                "best_val_loss": best_val,
                "total_steps": history[-1].get("step"),
                "convergence_speed": final_loss / history[-1].get("step") if final_loss else None
            }
        
        return run_stats
    
    @staticmethod
    def export_to_csv(metrics_file: str, output_csv: str):
        """Export metrics to CSV for external analysis"""
        import csv
        
        with open(metrics_file, 'r') as f:
            history = json.load(f)
        
        if not history:
            return
        
        # Get all keys
        all_keys = set()
        for entry in history:
            all_keys.update(entry.keys())
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(history)
        
        print(f"📄 Exported metrics to: {output_csv}")


# Example usage
if __name__ == "__main__":
    # Example: Track metrics during training
    tracker = ValidationLossTracker(
        output_dir="./outputs/test_run",
        validation_every_n_steps=50,
        patience=5
    )
    
    # Simulate training
    for step in range(1, 501):
        train_loss = 0.5 / (1 + step/100)  # Simulated decreasing loss
        
        tracker.log_metrics(
            step=step,
            train_loss=train_loss,
            learning_rate=0.0001 * (1 - step/500)
        )
        
        # Simulate validation
        if tracker.should_validate(step):
            val_loss = train_loss * 1.05  # Simulated val loss
            tracker.log_metrics(step=step, train_loss=train_loss, val_loss=val_loss)
            
            if tracker.check_early_stopping():
                print("Early stopping triggered!")
                break
    
    # Print final statistics
    print("\n" + "="*60)
    print("Final Statistics:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate plots
    tracker.plot_loss_curves()

