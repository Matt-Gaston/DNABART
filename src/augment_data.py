import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


class DNACorruptor:
    """Class to handle DNA sequence corruption operations."""
    
    VALID_NUCLEOTIDES = {'A', 'T', 'C', 'G'}
    
    @staticmethod
    def substitute_nucleotides(sequence: str, corruption_rate: float = 0.20) -> str:
        """
        Randomly substitute nucleotides in a DNA sequence.
        
        Args:
            sequence (str): The DNA sequence (e.g., "ATCG")
            corruption_rate (float): Fraction of nucleotides to corrupt (0 <= rate <= 1)
            
        Returns:
            str: Corrupted DNA sequence
        
        Raises:
            ValueError: If corruption_rate is not between 0 and 1
            ValueError: If sequence contains invalid nucleotides
        """
        if not 0 <= corruption_rate <= 1:
            raise ValueError("Corruption rate must be between 0 and 1")
        
        # Validate sequence
        invalid_chars = set(sequence) - DNACorruptor.VALID_NUCLEOTIDES
        if invalid_chars:
            raise ValueError(f"Invalid nucleotides found: {invalid_chars}")
        
        sequence_list = list(sequence)
        num_to_corrupt = int(len(sequence) * corruption_rate)
        
        # Get indices to corrupt
        corrupt_indices = random.sample(range(len(sequence)), num_to_corrupt)
        
        # Perform substitutions
        for idx in corrupt_indices:
            original = sequence_list[idx]
            substitutes = list(DNACorruptor.VALID_NUCLEOTIDES - {original})
            sequence_list[idx] = random.choice(substitutes)
        
        return ''.join(sequence_list)
    
    
    @staticmethod
    def delete_nucleotides(sequence: str, corruption_rate: float = 0.20) -> str:
        """
        Randomly delete nucleotides in a DNA sequence.
        
        Args:
            sequence (str): The DNA sequence (e.g., "ATCG")
            corruption_rate (float): Fraction of nucleotides to delete (0 <= rate <= 1)
            
        Returns:
            str: DNA sequence with deletions
        
        Raises:
            ValueError: If corruption_rate is not between 0 and 1
            ValueError: If sequence contains invalid nucleotides
        """
        if not 0 <= corruption_rate <= 1:
            raise ValueError("Corruption rate must be between 0 and 1")
        
        # Validate sequence
        invalid_chars = set(sequence) - DNACorruptor.VALID_NUCLEOTIDES
        if invalid_chars:
            raise ValueError(f"Invalid nucleotides found: {invalid_chars}")
        
        sequence_list = list(sequence)
        num_to_delete = int(len(sequence) * corruption_rate)
        
        # Don't delete everything
        if num_to_delete >= len(sequence):
            num_to_delete = len(sequence) - 1
        
        # Get indices to delete
        delete_indices = set(random.sample(range(len(sequence)), num_to_delete))
        
        # Create new sequence excluding deleted positions
        corrupted = ''.join(char for i, char in enumerate(sequence) if i not in delete_indices)
        
        return corrupted



class DatasetAugmenter:
    """Class to handle dataset augmentation operations."""
    
    def __init__(self, corruption_type: str = 'sub', corruption_rate: float = 0.1):
        """
        Initialize the augmenter.
        
        Args:
            corruption_rate (float): Rate of corruption to apply
        """
        self.corruptor = DNACorruptor()
        self.corruption_rate = corruption_rate
        self.corruption_type = corruption_type
        self.num_processes = max(1, cpu_count() -1) # Leave one cpu free
        
    
    def corrupt_sequence(self, sequence: str) -> str:
        """Helper function for parallel processing."""
        if self.corruption_type == 'sub':
            return self.corruptor.substitute_nucleotides(sequence, self.corruption_rate)
        elif self.corruption_type == 'del':
            return self.corruptor.delete_nucleotides(sequence, self.corruption_rate)
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")
    
    
    def process_file(self, input_path: Path) -> None:
        """
        Process a single file and create its corrupted version.
        
        Args:
            input_path (Path): Path to the input file
        """
        # Create output path
        output_path = input_path.parent / f'corrupted_{input_path.name}'
        
        # Read sequences
        with open(input_path, "r") as f:
            sequences = [line.strip() for line in tqdm(f, desc="Reading file", unit=" lines")]
        
        # Corrupt sequences using parallel processing
        with Pool(processes=self.num_processes) as pool:
            corrupted_sequences = list(tqdm(
                pool.imap(self.corrupt_sequence, sequences),
                total=len(sequences),
                desc="Corrupting sequences",
                unit=" seqs"
            ))
        
        # Write output
        with open(output_path, "w") as f:
            for sequence in tqdm(corrupted_sequences, desc="Writing output", unit=" lines"):
                f.write(f"{sequence}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DNA sequence dataset augmentation tool')
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
                        help='Path(s) to input dataset file(s)')
    # parser.add_argument('--out_dir', type=str)
    parser.add_argument('--cor_type', type=str, default='sub', choices=['sub'],
                        help='Corruption type to apply: substitution(sub)')
    parser.add_argument('--cor_rate', type=float, default=0.1,
                        help='Corruption rate (between 0 and 1)')
    
    args = parser.parse_args()
    
    augmenter = DatasetAugmenter(corruption_type=args.cor_type, corruption_rate=args.cor_rate)
    
    
    for path_str in args.datasets:
        input_path = Path(path_str).resolve()
        if not input_path.exists():
            print(f"Warning: File not found: {input_path}")
            continue
        
        try:
            augmenter.process_file(input_path)
            print(f"Successfully processed: {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
    

if __name__=='__main__':
    main()