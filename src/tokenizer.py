from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

special_tokens = ["<pad>","<unk>", "<sep>", "<mask>", "<s>", "</s>"]


def get_special_tokens():
    special_tokens_dict = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "sep_token": "<sep>",
        "mask_token": "<mask>",
        "bos_token": "<s>",
        "eos_token": "</s>"}
    return special_tokens_dict


def dontUseThisBPE(data_paths : str, vocab_size : int):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit('\n')


    print("start training")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train(data_paths, trainer)

    print("finished training")
    print("saving model")
    tokenizer.save('../models/DNABART_BPE_4096.json')
    
    
    
def bl_bpe(data_paths : str, out_dir : str, vocab_size : int):
    """Byte level BPE tokenizer trainer

    Args:
        data_paths (str): path to the dataset being trained on
        out_dir (str): path to the directory for storing tokenizer models
        vocab_size (int): vocab size to generate
    """
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=data_paths, vocab_size=vocab_size, min_frequency=2, show_progress=True, special_tokens=special_tokens)

    tokenizer.save_model(out_dir)
    
    

def cl_bpe(data_paths : str, out_dir : str, vocab_size : int):
    """Byte level BPE tokenizer trainer

    Args:
        data_paths (str): path to the dataset being trained on
        out_dir (str): path to the directory for storing tokenizer models
        vocab_size (int): vocab size to generate
    """
    tokenizer = CharBPETokenizer()

    tokenizer.train(files=data_paths, vocab_size=vocab_size, min_frequency=2, show_progress=True, special_tokens=special_tokens, suffix='')

    tokenizer.save_model(out_dir)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BPE tokenizer parameters')
    parser.add_argument('--datasets', type=str, required=True, nargs='+')
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--bpe_type', type=str, default='bl', choices=('bl', 'cl'))
    parser.add_argument('--out_dir', type=str, default='../models')

    args = parser.parse_args()
    # print(Path(args.datasets[0]).resolve().parents[1] / 'models')
    datapaths = [Path(patharg).resolve() for patharg in args.datasets]
    
    out_path = Path(args.out_dir).resolve() 
    
    if args.bpe_type == 'bl':
        out_path = out_path / f'DNABART_BLBPE_{args.vocab_size}'
        if not Path.exists(out_path):
            print(f'creating new directory for model: DNABART_BLBPE_{args.vocab_size}')
            os.mkdir(out_path)
            
        bl_bpe(args.datasets, str(out_path), args.vocab_size)
        
    elif args.bpe_type == 'cl':
        out_path = out_path / f'DNABART_CLBPE_{args.vocab_size}'
        if not Path.exists(out_path):
            print(f'creating new directory for model: DNABART_CLBPE_{args.vocab_size}')
            os.mkdir(out_path)
            
        cl_bpe(args.datasets, str(out_path), args.vocab_size)
