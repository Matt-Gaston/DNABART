import torch
from tqdm import tqdm
from transformers import BartTokenizerFast

def preprocess_and_cache(ground_truth_file, corrupted_file, tokenizer, max_length, cache_file):
    print("Preprocessing and caching dataset...")
    data = []
    with open(ground_truth_file, "r") as gt_file, open(corrupted_file, "r") as c_file:
        for gt_line, c_line in tqdm(zip(gt_file, c_file), desc="Tokenizing", unit=" lines"):
            gt_tokenized = tokenizer(
                gt_line.strip(),
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            c_tokenized = tokenizer(
                c_line.strip(),
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            data.append({
                'input_ids': c_tokenized['input_ids'].squeeze(0),
                'labels': gt_tokenized['input_ids'].squeeze(0),
                'attention_mask': c_tokenized['attention_mask'].squeeze(0)
            })
    
    # Save tokenized data
    torch.save(data, cache_file)
    print(f"Dataset cached to {cache_file}")


tokenizer = BartTokenizerFast.from_pretrained('../models/DNABART_BLBPE_4096')

# preprocess_and_cache("../data/test.txt", "../data/corrupted_test.txt", tokenizer, max_length=512, cache_file="../data/test_cache.pt")
preprocess_and_cache("../data/train_part1.txt", "../data/corrupted_train_part1.txt", tokenizer, max_length=512, cache_file="../data/trainp1_cache.pt")
