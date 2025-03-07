import torch
from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    Trainer,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

import os


import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from dataset import IterableDNABARTDataset, DNABARTClassificationDataset
import dnabart_config
import utils



def pre_train_model(training_gt_file="../data/test.txt", training_cor_file="../data/corrupted_test.txt", eval_gt_file="../data/dev.txt", eval_cor_file="../data/dev.txt"):
    print("1. Initializing defaults")
    checkpointdir = '../models/model_checkpoints'
    os.makedirs(checkpointdir, exist_ok=True)
    print("1. Done")
    
    
    print("2. Initializing tokenizer")
    tokenizer = BartTokenizerFast.from_pretrained('../models/DNABART_BLBPE_4096')
    print("2. Done")
    
    
    print("3. Initializing Datasets")
    train_data = IterableDNABARTDataset(
        ground_truth_file=training_gt_file,
        corrupted_file=training_cor_file,
        tokenizer=tokenizer
        )
    eval_data = IterableDNABARTDataset(
        ground_truth_file=eval_gt_file,
        corrupted_file=eval_cor_file,
        tokenizer=tokenizer
    )
    print("3. Done")
    
    
    print("4. Initializing Model")
    dnabartcfg = dnabart_config.get_dnabart_config()
    model = BartForConditionalGeneration(dnabartcfg)
    print("4. Done")
    
    
    print("5. Setting up training arguments")
    training_args = dnabart_config.get_dnabart_pretraining_config()
    print("5. Done")
    
    
    print("6. Initializing Trainer")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=utils.compute_metrics_rogue if eval_data else None,
    )
    print("6. Done")
    
    
    print('7. Beginning Training')
    trainer.train()
    print('7. Done')
    
    
    print("8. Saving final model")
    trainer.save_model(os.path.join(checkpointdir, 'finalmodel'))
    print("8. Done")
    
    
    print("Training completed!")
    


def train_classification_model(
    pretrained_model_path, 
    train_data_path, 
    val_data_path,
    tokenizer_path,
    save_path,
):
    
    print("Initializing tokenizer")
    tokenizer = BartTokenizerFast.from_pretrained(tokenizer_path)
    print("Done")
    
    print("Initializing Datasets")
    # Prepare datasets
    train_dataset = DNABARTClassificationDataset(train_data_path, tokenizer)
    val_dataset = DNABARTClassificationDataset(val_data_path, tokenizer)
    
    num_classes = len(pd.read_csv(train_data_path)['label'].unique())
    print("Done")
    
    
    print("Loading foundational model weights")
    model = BartForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path,
        num_labels=num_classes,
        )
    print("Done")
    
    
    temp_paths = Path(train_data_path).resolve().parent.parts
    final_save_path = os.path.join(save_path, temp_paths[-2], temp_paths[-1])
    os.makedirs(final_save_path, exist_ok=True)
    
    print("Setting up trainer")
    training_args = dnabart_config.get_dnabart_classification_ft_config()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=utils.compute_metrics_accuracy,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    print("Done")
    
    
    print("starting training")
    trainer.train()
    print("Done")
    
    
    trainer.save_model(os.path.join(final_save_path, 'best_model'))
    
    best_metric = trainer.state.best_metric
    print(f"Best Accuracy: {best_metric}")
    
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Training script for DNABART')
    parser.add_argument('--train_phase', type = str, nargs=1, choices=('pretrain', 'finetune'))
    parser.add_argument('--train_gt_file', type = str)
    parser.add_argument('--train_cor_file', type = str)
    parser.add_argument('--eval_gt_file', type = str)
    parser.add_argument('--eval_cor_file', type = str)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--model_type', type=str, default='gen', choices=('gen', 'cls'))
    
    
    args = parser.parse_args()
    
    # datapaths = Path(args.dataset).resolve()

    if args.train_phase == 'pretrain': #pretrain training phase selected
        #check if all needed files exist
        if not Path(args.train_gt_file).exists():
            print("Training ground truth file does not exist")
        if not Path(args.train_cor_file).exists():
            print("Training corrupted file does not exist")
        if not Path(args.eval_gt_file).exists():
            print("Evaluation ground truth file does not exist")
        if not Path(args.eval_cor_file).exists():
            print("Evaluation corrupted file does not exist")

        #all files validated, begin pretraining
        pre_train_model(training_gt_file=args.train_gt_file, training_cor_file=args.train_cor_file, eval_gt_file=args.eval_gt_file, eval_cor_file=args.eval_cor_file)
    
    if args.train_phase == 'finetune':
        pass
        ##TODO add finetuning code
    
    # PRETRAINED_MODEL_PATH = '../models/model_checkps/finalmodel'
    # TOKENIZER_PATH = '../models/DNABART_BLBPE_4096'
    # TRAIN_DATA_PATH = '../data/GUE/virus/covid/train.csv'
    # VAL_DATA_PATH = '../data/GUE/virus/covid/test.csv'
    # NUM_CLASSES = 3  # Adjust based on your classification task
    # SAVE_PATH = '../models/fine_tuned/GUE'
    # os.makedirs(SAVE_PATH, exist_ok=True)
    # model = train_classification_model(
    #     pretrained_model_path=PRETRAINED_MODEL_PATH,
    #     train_data_path=TRAIN_DATA_PATH,
    #     val_data_path=VAL_DATA_PATH,
    #     tokenizer_path=TOKENIZER_PATH,
    #     save_path=SAVE_PATH
    # )
    return


if __name__=='__main__':
    main()