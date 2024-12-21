import torch
from transformers import BartTokenizer, BartTokenizerFast
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartConfig
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
import glob
import os
from model import DNABARTForClassification


import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from dataset import DNABARTDataset, CachedDNABARTDataset, IterableDNABARTDataset, DNABARTClassificationDataset
import dnabart_config


def save_checkpoint(model, optimizer, scheduler, epoch, batch_num, checkpoint_dir):
    """
    Save model checkpoint with optimizer and scheduler states
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_e{epoch}_{batch_num}.pt')
    
    torch.save({
        'epoch': epoch,
        'batch': batch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    
    # print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """
    Load model checkpoint with optional optimizer and scheduler states
    """
    if device is None:
        device = next(model.parameters()).device
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (checkpoint['epoch'], checkpoint['batch'])



def pre_train_model():
    resume_training = True
    start_epoch, start_batch = 0, 0
    
    print("1. Initializing defaults")
    batch_size = 128
    lr=5e-5
    weight_decay = 0.1
    num_epochs = 2
    
    DEVICE = dnabart_config.get_device()
    
    checkpointdir = '../models/model_checkps'
    print("1. Done")
    
    print("2. Initializing tokenizer")
    tokenizer = BartTokenizerFast.from_pretrained('../models/DNABART_BLBPE_4096')
    print("2. Done")
    
    print("3. Initializing Dataset and DataLoader")
    # train_data = DNABARTDataset(ground_truth_file='../data/train_part1.txt', corrupted_file='../data/corrupted_train_part1.txt', tokenizer=tokenizer)
    # dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    train_data = IterableDNABARTDataset(ground_truth_file='../data/train_part1.txt', corrupted_file='../data/corrupted_train_part1.txt', tokenizer=tokenizer)
    dataloader = DataLoader(train_data, batch_size=batch_size)
    print("3. Done")
    
    print("4. Initializing Model")
    dnabartcfg = dnabart_config.get_dnabart_config()
    model = BartForConditionalGeneration(dnabartcfg)
    print("Sending model to device")
    model.to(DEVICE)
    print("4. Done")
    
    print("5. Constructing training optimizers")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps
    )
    
    
    if resume_training:
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpointdir, 'checkpoint_epoch_*.pt')),
            key=os.path.getmtime
        )
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpointdir, 'checkpoint_epoch_e0_bn40000.pt') #checkpoint_files[-1]
            print(f'Resuming from checkpoint: {latest_checkpoint}')
            checkpoint = load_checkpoint(latest_checkpoint, model, optimizer, scheduler, device=DEVICE)
            start_epoch, start_batch = checkpoint
        else:
            print("No checkpoint found, starting from scratch")
            model.to(DEVICE)
    
    print("5. Done")
    
    
    # scaler = torch.amp.GradScaler()
    
    print('7. Beginning Training')
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        batch_bar = tqdm(enumerate(dataloader), desc='Batch Num', leave=False, position=1, total=len(dataloader))
        for batch_idx, batch in batch_bar:
            if epoch == start_epoch and batch_idx <= start_batch:
                continue
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            #mixed precision training, not working
            # with torch.amp.autocast('cuda'):
            #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            #     loss = outputs.loss
            
            # scaler.scale(loss.backward())
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()
            
            #NORMAL Stuff, without mixed precision forward pass training
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            batch_bar.set_description(f'Batch Num (Loss: {loss.item():.4f})')
            if batch_idx % 8000 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, checkpointdir)
        
    model.save_pretrained('../models/model_checkps/finalmodel')
    print('7. Done')
    

# tokenizer = BartTokenizer.from_pretrained('../models/DNABART_BLBPE_4096')


# def evaluate_model(model_path, tokenizer_path, test_dataset_in_path, test_dataset_labels_path):  
#     print("1. Initializing defaults")
#     batch_size = 32
    
#     DEVICE = dnabart_config.get_device()
    
#     tokenizer = BartTokenizerFast.from_pretrained(tokenizer_path)
    
#     model = BartForConditionalGeneration
    


def train_classification_model(
    pretrained_model_path, 
    train_data_path, 
    val_data_path,
    tokenizer_path,
    save_path,
    batch_size=10,
    learning_rate=2e-5,
    num_epochs=8
):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = BartTokenizerFast.from_pretrained(tokenizer_path)
    
    # Prepare datasets
    train_dataset = DNABARTClassificationDataset(train_data_path, tokenizer)
    val_dataset = DNABARTClassificationDataset(val_data_path, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(pd.read_csv(train_data_path)['label'].unique())
    
    # Initialize model
    model = DNABARTForClassification(pretrained_model_path, num_classes)
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc=f"Epochs", position=0):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        batch_bar = tqdm(enumerate(train_dataloader) , desc=f"Batch Num", leave=False, position=1, total=len(train_dataloader))
        for batch_idx, batch in batch_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            batch_bar.set_description(f'Batch Num (Loss: {loss.item():.4f})')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, logits = model(input_ids, attention_mask, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Print epoch results
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {train_loss/len(train_dataloader):.4f}")
        print(f"Training Accuracy: {100 * train_correct / train_total:.2f}%")
        print(f"Validation Loss: {val_loss/len(val_dataloader):.4f}")
        print(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            temp_paths = Path(train_data_path).resolve().parent.parts
            final_path = os.path.join(save_path,temp_paths[-2], temp_paths[-1])
            os.makedirs(final_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(final_path, 'best_class_model.pt'))
    print(f"Best Accuracy: {best_val_acc}")
    return model



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training script for DNABART')
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--model_type', type=str, default='gen', choices=('gen', 'cls'))
    
    args = parser.parse_args()
    
    # datapaths = Path(args.dataset).resolve()
    
    # pre_train_model()
    
    PRETRAINED_MODEL_PATH = '../models/model_checkps/finalmodel'
    TOKENIZER_PATH = '../models/DNABART_BLBPE_4096'
    TRAIN_DATA_PATH = '../data/GUE/virus/covid/train.csv'
    VAL_DATA_PATH = '../data/GUE/virus/covid/test.csv'
    NUM_CLASSES = 3  # Adjust based on your classification task
    SAVE_PATH = '../models/fine_tuned/GUE'
    os.makedirs(SAVE_PATH, exist_ok=True)
    model = train_classification_model(
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        train_data_path=TRAIN_DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        tokenizer_path=TOKENIZER_PATH,
        save_path=SAVE_PATH
    )