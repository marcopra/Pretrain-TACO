"""
Script for pretraining ResNet models on Tiny ImageNet dataset using supervised learning.
The saved models will be compatible with the existing TACO ResNet encoder code.

Usage examples:
python pretrain_resnet_tiny_imagenet.py --model resnet18 --epochs 100 --lr 0.001 --batch_size 256
python pretrain_resnet_tiny_imagenet.py --model resnet50 --epochs 200 --lr 0.0001 --batch_size 128
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import wandb
from pathlib import Path
import time

def create_tiny_imagenet_transforms():
    """Create transforms for Tiny ImageNet (64x64 -> 224x224 for ResNet)"""
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_tiny_imagenet(data_dir, batch_size, num_workers=4):
    """Load Tiny ImageNet dataset"""
    # train_transform, val_transform = create_tiny_imagenet_transforms()
    
    # Load training data
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        # transform=train_transform
    )
    
    # Load validation data
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        # transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)

def create_resnet_model(model_name, num_classes):
    """Create ResNet model"""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights=None)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Modify final layer for Tiny ImageNet classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def save_model_checkpoint(model, save_path, model_name, epoch, train_acc, val_acc, args):
    """Save model checkpoint in format compatible with existing code"""
    os.makedirs(save_path, exist_ok=True)
    
    # Create filename
    filename = f"{model_name}_tiny_imagenet_ep{epoch}_train{train_acc:.1f}_val{val_acc:.1f}.pt"
    filepath = os.path.join(save_path, filename)
    
    # Save full model state dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'model_name': model_name,
        'args': vars(args),
        'num_classes': 200  # Tiny ImageNet has 200 classes
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to: {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Pretrain ResNet on Tiny ImageNet')
    parser.add_argument('--data_dir', type=str, default='/path/to/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset')
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet50'], default='resnet18',
                        help='ResNet model to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--save_path', type=str, default='models/',
                        help='Path to save trained models')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='resnet-tiny-imagenet-pretrain',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WandB run name')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Tiny ImageNet dataset not found at {args.data_dir}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb_config = {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "dataset": "tiny-imagenet-200",
            "device": str(device)
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )
    
    # Load dataset
    print("Loading Tiny ImageNet dataset...")
    train_loader, val_loader, num_classes = load_tiny_imagenet(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_resnet_model(args.model, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        print('-' * 50)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            })
        
        # Save checkpoint periodically and if best validation accuracy
        if (epoch + 1) % args.save_every == 0 or val_acc > best_val_acc:
            save_model_checkpoint(model, args.save_path, args.model, 
                                epoch + 1, train_acc, val_acc, args)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_path = save_model_checkpoint(model, args.save_path, args.model, 
                                     args.epochs, train_acc, val_acc, args)
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final model saved to: {final_path}")
    
    if args.use_wandb:
        wandb.log({"best_val_accuracy": best_val_acc})
        wandb.finish()

if __name__ == '__main__':
    main()
