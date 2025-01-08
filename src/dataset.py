import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import pandas as pd

class DNABARTDataset(Dataset):
    def __init__(self, ground_truth_file, corrupted_file, tokenizer, max_length=512):
        """
        Custom dataset for DNA sequences with ground truth and corrupted data.

        Args:
            ground_truth_file (str): Path to the file containing ground truth DNA sequences.
            corrupted_file (str): Path to the file containing corrupted DNA sequences.
            
            max_length (int): Max seq length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Opening target file")
        with open(ground_truth_file, "r") as gt_file:
            ground_truth_sequences = []
            for line in tqdm(gt_file, desc="Reading target file", unit=" lines"):
                ground_truth_sequences.append(line.strip())
            # ground_truth_sequences = [line.strip() for line in gt_file.readlines()]
            temp = self.tokenizer(ground_truth_sequences, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            self.labels = temp['input_ids']
            self.attention_masks =temp['attention_mask']
            del ground_truth_sequences
        print("Done processing target file")
        print("Opening corrupted file")
        with open(corrupted_file, "r") as c_file:
            corrupted_sequences = []
            for line in tqdm(c_file, desc="Reading corrupted file", unit=" lines"):
                corrupted_sequences.append(line.strip())
            self.inputs = self.tokenizer(corrupted_sequences, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
            del corrupted_sequences
        print("Done processing corrupted file")
        
        # Ensure the two files have the same number of sequences
        assert len(self.labels) == len(self.inputs), \
            "Mismatch between ground truth and corrupted file lengths."
            
        
        
        # temp = self.tokenizer(ground_truth_sequences, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        # self.labels = temp['input_ids']
        # self.inputs = self.tokenizer(corrupted_sequences, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        # self.attention_masks =temp['attention_mask']
        
        
        
        # self.tokenized_data = []
        # for ground_truth, corrupted in zip(ground_truth_sequences, corrupted_sequences):
        #     corrupted_tokens = self.tokenizer(corrupted, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        #     ground_truth_tokens = self.tokenizer(ground_truth, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        #     self.tokenized_data.append({'input':corrupted_tokens['input_ids'].squeeze(0),
        #                                 'attention_mask':corrupted_tokens['attention_mask'].squeeze(0),
        #                                 'label':ground_truth_tokens['input_ids'].squeeze(0)})

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve a ground truth and corrupted sequence pair by index.

        Args:
            idx (int): Index of the sequence pair to retrieve.

        Returns:
            dict: A dictionary with 'ground_truth' and 'corrupted' sequences.
        """
        return {
            'input_ids' : self.inputs[idx],
            'labels' : self.labels[idx],
            'attention_mask' : self.attention_masks[idx]
        }



class CachedDNABARTDataset(Dataset):
    def __init__(self, cache_file):
        self.data = torch.load(cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class IterableDNABARTDataset(IterableDataset):
    def __init__(self, ground_truth_file, corrupted_file, tokenizer, max_length=256):
        """
        Iterable dataset for DNA sequences with ground truth and corrupted data.

        Args:
            ground_truth_file (str): Path to the file containing ground truth DNA sequences.
            corrupted_file (str): Path to the file containing corrupted DNA sequences.
            tokenizer (BartTokenizerFast): Tokenizer for the sequences.
            max_length (int): Maximum sequence length.
        """
        self.ground_truth_file = ground_truth_file
        self.corrupted_file = corrupted_file
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        """
        Stream data from both files and yield tokenized samples.

        Yields:
            dict: A dictionary with tokenized inputs, labels, and attention masks.
        """
        with open(self.ground_truth_file, "r") as f_gt, open(self.corrupted_file, "r") as f_cor:
            for gt_line, cor_line in zip(f_gt, f_cor):
                # Tokenize the sequences
                input_tokens = self.tokenizer(
                    cor_line.strip(),
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                label_tokens = self.tokenizer(
                    gt_line.strip(),
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                # Yield a single sample
                yield {
                    'input_ids': input_tokens['input_ids'].squeeze(),
                    'attention_mask': input_tokens['attention_mask'].squeeze(),
                    'labels': label_tokens['input_ids'].squeeze()
                }
    def __len__(self):
        """
        Count total number of sequences in the files.
        """
        with open(self.ground_truth_file, 'r') as f:
            return sum(1 for _ in f)
        


class DNABARTClassificationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        """
        Initialize the dataset for classification
        
        :param file_path: Path to the dataset file
        :param tokenizer: Tokenizer to use
        :param max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        df = pd.read_csv(csv_path)
    
        if 'sequence' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must contain 'sequence' and 'label' columns")
            
        # Validate and convert labels to integers if needed
        if df['label'].dtype == object:
            # If labels are strings, create a label mapping
            unique_labels = df['label'].unique()
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.data = [(row['sequence'], self.label_map[row['label']]) for _, row in df.iterrows()]
            print("Label Mapping:", self.label_map)
        else:
            # If labels are already numeric
            self.data = [(row['sequence'], row['label']) for _, row in df.iterrows()]
            self.label_map = None
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }