#!/usr/bin/env python
"""
Use Best TCN Ray Tune Configuration
Utility script to load and use the best hyperparameters from Ray Tune for final TCN model training.
"""

import json
import torch
from pathlib import Path
import sys
from functools import partial

# Add training path for imports
sys.path.append('training')

from ml_pipeline.tcn_tune_ray import (
    train_tcn_with_tune, 
    Seq2SeqTCN, 
    load_vocab_and_paths, 
    get_device,
    define_spike_threshold,
    calculate_spike_recall,
    masked_mse
)
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed
from torch.utils.data import DataLoader

def load_best_tcn_configs():
    """Load the best TCN configurations found by Ray Tune."""
    config_path = Path("ml_pipeline/results/best_tcn_configs.json")
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Best TCN configs not found at {config_path}. "
            "Run 'python -m ml_pipeline.run_tcn_hpo' first."
        )
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return data

def train_with_best_config(config_name="best_loss", epochs=50, save_model=True):
    """
    Train TCN model using the best configuration from Ray Tune results.
    
    Args:
        config_name: Either "best_loss" or "best_recall" 
        epochs: Number of epochs to train
        save_model: Whether to save model checkpoint
    """
    
    print(f"Loading best TCN configuration: {config_name}")
    best_configs = load_best_tcn_configs()
    
    if config_name not in best_configs:
        raise ValueError(f"Config '{config_name}' not found. Available: {list(best_configs.keys())}")
    
    config = best_configs[config_name]["config"]
    metrics = best_configs[config_name]["metrics"]
    
    print("Best configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nBest metrics achieved:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTraining TCN model with best {config_name} configuration...")
    print("="*80)
    
    # Setup device and paths
    device = get_device()
    vocab, arrow_path = load_vocab_and_paths()
    
    # Constants
    HORIZON = 30
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
    val_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
    test_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
    
    # Create dataloaders with best batch size
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=HORIZON),
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(pad_and_mask_fixed, flag_kind="val", horizon=HORIZON),
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=HORIZON),
        num_workers=0
    )
    
    # Get feature dimensions from first batch
    first_batch = next(iter(train_loader))
    num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = first_batch
    
    num_features = num_tensor.shape[-1]
    bool_features = boo_tensor.shape[-1]
    cat_features_list = list(vocab.keys())
    
    print(f"Feature dimensions: {num_features} numeric, {bool_features} boolean, {len(cat_features_list)} categorical")
    
    # Define spike threshold from training data
    print("Computing spike threshold from training data...")
    epss_samples = []
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Sample first 10 batches
            break
        _, _, _, targets, _, _, _, _, _ = batch
        epss_samples.append(targets.cpu().numpy())
    
    epss_samples = torch.cat([torch.tensor(s) for s in epss_samples], dim=0)
    spike_threshold = define_spike_threshold(epss_samples.flatten().numpy())
    print(f"Spike threshold (95th percentile): {spike_threshold:.4f}")
    
    # Initialize model with best hyperparameters
    print("Initializing TCN model...")
    model = Seq2SeqTCN(
        vocab=vocab,
        num_features=num_features,
        bool_features=bool_features,
        cat_features=cat_features_list,
        nb_filters=config["nb_filters"],
        levels=config["levels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        horizon=HORIZON
    ).to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)
    
    model.apply(init_weights)
    
    # Setup optimizer and scheduler with best learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {epochs} epochs with lr={config['lr']:.2e}")
    print("="*80)
    
    # Training loop
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_spike_recall": []}
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = batch
            
            # Move to device
            num_tensor = num_tensor.to(device, non_blocking=True)
            boo_tensor = boo_tensor.to(device, non_blocking=True)
            cat_tensor = cat_tensor.to(device, non_blocking=False)
            target_tensor = target_tensor.to(device, non_blocking=True)
            m_t = m_t.to(device, non_blocking=True)
            m_eval = m_eval.to(device, non_blocking=True)
            m_h = m_h.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(num_tensor, boo_tensor, cat_tensor)
                    loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(num_tensor, boo_tensor, cat_tensor)
                loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_spike_recall = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = val_batch
                
                num_tensor = num_tensor.to(device, non_blocking=True)
                boo_tensor = boo_tensor.to(device, non_blocking=True)
                cat_tensor = cat_tensor.to(device, non_blocking=False)
                target_tensor = target_tensor.to(device, non_blocking=True)
                m_t = m_t.to(device, non_blocking=True)
                m_eval = m_eval.to(device, non_blocking=True)
                m_h = m_h.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        pred = model(num_tensor, boo_tensor, cat_tensor)
                        loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
                else:
                    pred = model(num_tensor, boo_tensor, cat_tensor)
                    loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
                
                # Calculate spike recall
                pred_flat = pred.cpu().numpy().flatten()
                target_flat = target_tensor.cpu().numpy().flatten()
                batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
                
                val_loss += loss.item()
                val_spike_recall += batch_spike_recall
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches
        avg_val_spike_recall = val_spike_recall / val_batches
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_spike_recall"].append(avg_val_spike_recall)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Spike Recall: {avg_val_spike_recall:.3f}")
        
        model.train()
    
    # Final test evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    model.eval()
    test_loss = 0.0
    test_spike_recall = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for test_batch in test_loader:
            num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = test_batch
            
            num_tensor = num_tensor.to(device, non_blocking=True)
            boo_tensor = boo_tensor.to(device, non_blocking=True)
            cat_tensor = cat_tensor.to(device, non_blocking=False)
            target_tensor = target_tensor.to(device, non_blocking=True)  
            m_t = m_t.to(device, non_blocking=True)
            m_eval = m_eval.to(device, non_blocking=True)
            m_h = m_h.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(num_tensor, boo_tensor, cat_tensor)
                    loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
            else:
                pred = model(num_tensor, boo_tensor, cat_tensor)
                loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
            
            # Calculate spike recall
            pred_flat = pred.cpu().numpy().flatten()
            target_flat = target_tensor.cpu().numpy().flatten()
            batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
            
            test_loss += loss.item()
            test_spike_recall += batch_spike_recall
            test_batches += 1
    
    final_test_loss = test_loss / test_batches
    final_test_spike_recall = test_spike_recall / test_batches
    
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Spike Recall: {final_test_spike_recall:.3f}")
    
    # Save results
    if save_model:
        print("\n" + "="*80)
        print("SAVING MODEL AND RESULTS")
        print("="*80)
        
        # Ensure results directory exists
        results_dir = Path("ml_pipeline/results")
        results_dir.mkdir(exist_ok=True)
        
        # Save model checkpoint with best config info
        checkpoint = {
            "config": config,
            "tuning_metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_history": history,
            "final_test_loss": final_test_loss,
            "final_test_spike_recall": final_test_spike_recall,
            "spike_threshold": spike_threshold,
            "epochs_trained": epochs
        }
        
        checkpoint_path = results_dir / f"best_tcn_{config_name}_final.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Model checkpoint saved: {checkpoint_path}")
        
        # Save training history
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_path = results_dir / f"tcn_{config_name}_training_history.csv"
        history_df.to_csv(history_path, index=False)
        print(f"✓ Training history saved: {history_path}")
        
        print(f"✓ Final test loss: {final_test_loss:.4f}")
        print(f"✓ Final test spike recall: {final_test_spike_recall:.3f}")
    
    return model, history, final_test_loss, final_test_spike_recall

def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TCN with best Ray Tune configuration")
    parser.add_argument("--config", choices=["best_loss", "best_recall"], default="best_loss",
                       help="Which best configuration to use")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs to train")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save model checkpoint")
    
    args = parser.parse_args()
    
    try:
        model, history, test_loss, test_recall = train_with_best_config(
            config_name=args.config,
            epochs=args.epochs,
            save_model=not args.no_save
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Used configuration: {args.config}")
        print(f"Final test loss: {test_loss:.4f}")
        print(f"Final test spike recall: {test_recall:.3f}")
        
        if not args.no_save:
            print(f"Model saved in: ml_pipeline/results/")
            print("Next steps:")
            print("1. Review training history CSV")
            print("2. Load checkpoint for inference")
            print("3. Generate predictions on new data")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 