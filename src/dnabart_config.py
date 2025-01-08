import torch
from transformers import BartConfig
from transformers import Seq2SeqTrainingArguments, TrainingArguments

import os


#directory configs
checkpoint_dir = '../models/model_checkpoints'
models_dir = '../models'



def get_dnabart_config():
    """
    Returns a customized BartConfig for DNABART.

    Returns:
        BartConfig: Customized configuration object.
    """    
    config = BartConfig(
        vocab_size=4096,            # Adjust based on your tokenizer
        d_model=768,                # Hidden size of the encoder and decoder
        encoder_layers=6,           # Number of encoder layers
        decoder_layers=6,           # Number of decoder layers
        encoder_attention_heads=12, # Number of attention heads in the encoder
        decoder_attention_heads=12, # Number of attention heads in the decoder
        encoder_ffn_dim=3072,       # Feed-forward network dimension in encoder
        decoder_ffn_dim=3072,       # Feed-forward network dimension in decoder
        max_position_embeddings=512, # Maximum position embeddings
        dropout=0.1,                # Dropout rate
        attention_dropout=0.1,      # Dropout rate for attention layers
        activation_function="gelu", # Activation function
        pad_token_id=0,
        unk_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        bos_token_id=4,
        eos_token_id=5
    )
    return config


def get_dnabart_pretraining_config():
    training_args = Seq2SeqTrainingArguments(
        # Basic Training Parameters
        output_dir=checkpoint_dir,
        num_train_epochs=2,
        per_device_train_batch_size=100,
        per_device_eval_batch_size=512,
        learning_rate=5e-5,
        weight_decay=0.1,
        
        # Training and Evaluation Flags
        do_train=True,
        do_eval=True,
        
        # Evaluation Strategy
        eval_strategy="steps",
        eval_steps=500,  # Run evaluation every 1000 steps
        eval_delay=0,  # Start evaluation right from the beginning
        
        # Seq2Seq Specific Parameters
        predict_with_generate=True,  # Use generate() for evaluation
        # generation_max_length=128,   # Max length for generated sequences
        generation_num_beams=4,      # Number of beams for beam search
        
        # Optimization Parameters
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        
        # Mixed Precision Training
        fp16=torch.cuda.is_available(),
        fp16_opt_level="O1",
        half_precision_backend="auto",
        
        # Checkpointing
        save_strategy="steps",
        save_steps=1000,  # Save at the same frequency as evaluation
        save_total_limit=3,  # Keep only the last 3 checkpoints
        resume_from_checkpoint=True,
        
        # Model Selection Based on Evaluation
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better
        
        # Logging for Tensorboard/Weights & Biases
        logging_dir=os.path.join(checkpoint_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        report_to=["tensorboard"],  # Can add "wandb" if using Weights & Biases
        
        # Memory Optimization
        # gradient_checkpointing=False,
        optim="adamw_torch",
        
        # Performance
        # group_by_length=True,
        # length_column_name="length",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        
        # Distributed Training
        # local_rank=-1,
        # ddp_find_unused_parameters=False,
        
        # Additional Parameters
        remove_unused_columns=True,
        label_names=["labels"],  # Important for seq2seq tasks
        
        # Sorting & Sampling
        sortish_sampler=True,  # Helps with training efficiency for seq2seq
        
        # Debugging
        debug="",  # Set to "underflow_overflow" for debugging
        
        # Push to Hub settings (optional)
        push_to_hub=False,
    )
    return training_args


def get_dnabart_classification_ft_config():
    
    training_args = TrainingArguments(
        #Basic training parameters
        output_dir=None,
        num_train_epochs=8,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        
        # Save best model
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        
        # Logging
        logging_dir=os.path.join(models_dir, 'logs'),
        logging_steps=50,
        report_to=["tensorboard"],
        
        # Optimizer settings
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        # gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        
        # Other settings
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        
        # Push to Hub settings (optional)
        push_to_hub=False,
    )
    
    return training_args

def get_device():
    """
    Returns the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        print("Using cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")