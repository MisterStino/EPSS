#!/usr/bin/env python
"""
Use Best Ray Tune Configuration
Utility script to load and use the best hyperparameters from Ray Tune for final model training.
"""

import json
import torch
from pathlib import Path
import sys

# Add training path for imports
sys.path.append('training')

from ml_pipeline.lstm_tune_ray import (
    train_lstm_with_tune, 
    Seq2SeqLSTM, 
    load_vocab_and_paths, 
    get_device,
    define_spike_threshold,
    calculate_spike_recall,
    masked_mse
)
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed
from torch.utils.data import DataLoader

def load_best_configs():
    """Load the best configurations found by Ray Tune."""
    config_path = Path("ml_pipeline/results/best_configs.json")
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Best configs not found at {config_path}. "
            "Run 'python -m ml_pipeline.lstm_tune_ray' first."
        )
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return data

def train_final_model(config, objective="loss", save_path=None):
    """
    Train the final model using the best configuration.
    
    Args:
        config: Configuration dictionary
        objective: Which objective to optimize for ("loss" or "spike_recall")
        save_path: Path to save the final model
    """
    
    print(f"Training final model with best config for {objective}...")
    print(f"Configuration: {config}")
    
    # Setup
    device = get_device()
    vocab, arrow_path = load_vocab_and_paths()
    
    # Create datasets
    train_dataset = CVEIterableDatasetFixed(arrow_path, horizon=30)
    val_dataset = CVEIterableDatasetFixed(arrow_path, horizon=30)
    test_dataset = CVEIterableDatasetFixed(arrow_path, horizon=30)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=30),
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        collate_fn=partial(pad_and_mask_fixed, flag_kind="val", horizon=30),
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=30),
        num_workers=0
    )
    
    # Get feature dimensions
    first_batch = next(iter(train_loader))
    num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_eval, m_h = first_batch
    
    num_features = num_tensor.shape[-1]
    bool_features = boo_tensor.shape[-1]
    cat_features_list = list(vocab.keys())
    
    # Define spike threshold
    epss_samples = []
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        _, _, _, targets, _, _, _ = batch
        epss_samples.append(targets.cpu().numpy())
    
    import numpy as np
    epss_samples = np.concatenate(epss_samples, axis=0)
    spike_threshold = define_spike_threshold(epss_samples.flatten())
    
    print(f"Spike threshold (95th percentile): {spike_threshold:.4f}")
    
    # Initialize model
    model = Seq2SeqLSTM(
        vocab=vocab,
        num_features=num_features,
        bool_features=bool_features,
        cat_features=cat_features_list,
        hidden_size=config["hidden_size"],
        layers=config["layers"],
        dropout=config["dropout"],
        horizon=30
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    best_val_spike_recall = 0.0
    
    for epoch in range(config["epochs"]):
        # Training
        epoch_loss = 0.0
        epoch_spike_recall = 0.0
        num_batches = 0
        
        for batch in train_loader:
            num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_eval, m_h = batch
            
            # Move to device
            num_tensor = num_tensor.to(device)
            boo_tensor = boo_tensor.to(device)
            cat_tensor = cat_tensor.to(device)
            target_tensor = target_tensor.to(device)
            m_t = m_t.to(device)
            m_eval = m_eval.to(device)
            m_h = m_h.to(device)
            
            # Forward pass
            pred = model(num_tensor, boo_tensor, cat_tensor)
            loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
            
            # Calculate spike recall
            with torch.no_grad():
                pred_flat = pred.cpu().numpy().flatten()
                target_flat = target_tensor.cpu().numpy().flatten()
                batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_spike_recall += batch_spike_recall
            num_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_spike_recall = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_eval, m_h = val_batch
                
                # Move to device
                num_tensor = num_tensor.to(device)
                boo_tensor = boo_tensor.to(device)
                cat_tensor = cat_tensor.to(device)
                target_tensor = target_tensor.to(device)
                m_t = m_t.to(device)
                m_eval = m_eval.to(device)
                m_h = m_h.to(device)
                
                # Forward pass
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
        avg_train_spike_recall = epoch_spike_recall / num_batches
        avg_val_loss = val_loss / val_batches
        avg_val_spike_recall = val_spike_recall / val_batches
        
        # Step scheduler
        scheduler.step()
        
        # Track best validation metrics
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        if avg_val_spike_recall > best_val_spike_recall:
            best_val_spike_recall = avg_val_spike_recall
        
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.6f}, Train Spike Recall: {avg_train_spike_recall:.4f}")
        print(f"  Val Loss: {avg_val_loss:.6f}, Val Spike Recall: {avg_val_spike_recall:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.8f}")
        
        model.train()
    
    # Test evaluation
    model.eval()
    test_loss = 0.0
    test_spike_recall = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for test_batch in test_loader:
            num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_eval, m_h = test_batch
            
            # Move to device
            num_tensor = num_tensor.to(device)
            boo_tensor = boo_tensor.to(device)
            cat_tensor = cat_tensor.to(device)
            target_tensor = target_tensor.to(device)
            m_t = m_t.to(device)
            m_eval = m_eval.to(device)
            m_h = m_h.to(device)
            
            # Forward pass
            pred = model(num_tensor, boo_tensor, cat_tensor)
            loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
            
            # Calculate spike recall
            pred_flat = pred.cpu().numpy().flatten()
            target_flat = target_tensor.cpu().numpy().flatten()
            batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
            
            test_loss += loss.item()
            test_spike_recall += batch_spike_recall
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches
    avg_test_spike_recall = test_spike_recall / test_batches
    
    print("\n" + "="*60)
    print("FINAL MODEL TRAINING COMPLETED")
    print("="*60)
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Best Validation Spike Recall: {best_val_spike_recall:.4f}")
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Spike Recall: {avg_test_spike_recall:.4f}")
    
    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab': vocab,
            'spike_threshold': spike_threshold,
            'best_val_loss': best_val_loss,
            'best_val_spike_recall': best_val_spike_recall,
            'test_loss': avg_test_loss,
            'test_spike_recall': avg_test_spike_recall,
            'model_architecture': {
                'num_features': num_features,
                'bool_features': bool_features,
                'cat_features': cat_features_list
            }
        }, save_path)
        
        print(f"Model saved to: {save_path}")
    
    return model, {
        'best_val_loss': best_val_loss,
        'best_val_spike_recall': best_val_spike_recall,
        'test_loss': avg_test_loss,
        'test_spike_recall': avg_test_spike_recall
    }

def main():
    """Main function to train final models using best configurations."""
    
    # Load best configurations
    try:
        best_configs = load_best_configs()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run hyperparameter optimization first:")
        print("python -m ml_pipeline.lstm_tune_ray")
        return
    
    print("Available best configurations:")
    print("1. Best for validation loss")
    print("2. Best for spike recall")
    
    # Train model optimized for loss
    print("\n" + "="*80)
    print("TRAINING MODEL OPTIMIZED FOR VALIDATION LOSS")
    print("="*80)
    
    loss_config = best_configs["best_for_loss"]["config"]
    model_loss, metrics_loss = train_final_model(
        config=loss_config,
        objective="loss",
        save_path="ml_pipeline/results/best_model_for_loss.pth"
    )
    
    # Train model optimized for spike recall
    print("\n" + "="*80)
    print("TRAINING MODEL OPTIMIZED FOR SPIKE RECALL")
    print("="*80)
    
    spike_config = best_configs["best_for_spike_recall"]["config"]
    model_spike, metrics_spike = train_final_model(
        config=spike_config,
        objective="spike_recall",
        save_path="ml_pipeline/results/best_model_for_spike_recall.pth"
    )
    
    # Summary
    print("\n" + "="*80)
    print("FINAL TRAINING SUMMARY")
    print("="*80)
    
    print("\nModel optimized for LOSS:")
    print(f"  Test Loss: {metrics_loss['test_loss']:.6f}")
    print(f"  Test Spike Recall: {metrics_loss['test_spike_recall']:.4f}")
    
    print("\nModel optimized for SPIKE RECALL:")
    print(f"  Test Loss: {metrics_spike['test_loss']:.6f}")
    print(f"  Test Spike Recall: {metrics_spike['test_spike_recall']:.4f}")
    
    # Determine which model is better overall
    if metrics_loss['test_spike_recall'] > metrics_spike['test_spike_recall'] and \
       metrics_loss['test_loss'] < metrics_spike['test_loss']:
        print("\nRecommendation: Use the model optimized for LOSS (better on both metrics)")
    elif metrics_spike['test_spike_recall'] > metrics_loss['test_spike_recall'] and \
         metrics_spike['test_loss'] < metrics_loss['test_loss']:
        print("\nRecommendation: Use the model optimized for SPIKE RECALL (better on both metrics)")
    else:
        print("\nRecommendation: Choose model based on your priority:")
        print("  - For general EPSS forecasting: use loss-optimized model")
        print("  - For high-risk spike detection: use spike-recall-optimized model")

if __name__ == "__main__":
    main() 