"""
Training script for weather-adaptive image dehazing.

Requirements:
    - CUDA-capable GPU (RTX 3050 or better)
    - Paired hazy/clear dataset in data/ directory
    
Usage:
    python train.py --data_dir ../data --epochs 10 --batch_size 4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for hazy, clear, condition in pbar:
        # Move to GPU
        hazy = hazy.to(device)
        clear = clear.to(device)
        condition = condition.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(hazy, condition)
        
        # Compute L1 loss
        loss = criterion(output, clear)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for hazy, clear, condition in tqdm(dataloader, desc='Validation'):
            hazy = hazy.to(device)
            clear = clear.to(device)
            condition = condition.to(device)
            
            output = model(hazy, condition)
            loss = criterion(output, clear)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train dehazing model')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (max 50)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (2-4 recommended for RTX 3050)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model weights')
    args = parser.parse_args()
    
    # Enforce CUDA requirement
    assert torch.cuda.is_available(), \
        "CUDA is required! Please ensure you have a CUDA-capable GPU."
    
    # Explicitly verify GPU
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    print(f"Using device: {device}")
    print(f"Active GPU: {device_name}")
    
    if "3050" not in device_name:
        print("WARNING: Active GPU does not appear to be RTX 3050. Please check CUDA_VISIBLE_DEVICES.")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading dataset from {args.data_dir}...")
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = get_model(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'dehaze_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✓ Best model saved to {save_path}")
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'dehaze_model_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1],
    }, final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses,
                        save_path=os.path.join(args.save_dir, 'training_curves.png'))
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
