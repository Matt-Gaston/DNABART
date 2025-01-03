import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to perform random substitutions
def random_substitution(sequence, corruption_rate=0.30):
    """
    Randomly substitute nucleotides in a DNA sequence.

    Args:
        sequence (str): The DNA sequence (e.g., "ATCG").
        corruption_rate (float): Fraction of nucleotides to corrupt (0 <= rate <= 1).

    Returns:
        str: Corrupted DNA sequence.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = list(sequence)  # Convert to list for mutability
    num_to_corrupt = int(len(sequence) * corruption_rate)
    corrupt_indices = random.sample(range(len(sequence)), num_to_corrupt)
    
    for idx in corrupt_indices:
        original = sequence[idx]
        substitutes = [n for n in nucleotides if n != original]
        sequence[idx] = random.choice(substitutes)
    
    return ''.join(sequence)


def augment_dataset(datapath:str):
    # Load and process your dataset
    file_path = "path/to/your/dna_sequences.txt"

    with open(datapath, "r") as f:
        sequences = list(tqdm(f, desc="Reading File", unit=" lines"))
    print("Done Reading")

    corrupted_sequences = []
    for seq in tqdm(sequences, desc="Corrupting sequences", unit=" seqs"):
        corrupted_sequences.append(random_substitution(seq.strip()))
    # corrupted_sequences = [None] * len(sequences)
    # with ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(random_substitution, seq.strip()): idx for idx, seq in enumerate(sequences)}
    #     with tqdm(total=len(futures), desc="Corrupting sequences", unit=" seqs") as pbar:
    #         for future in as_completed(futures):
    #             idx = futures[future]
    #             corrupted_sequences[idx] = future.result()
    #             pbar.update(1)  # Manually update the progress bar
    print("Done augmenting")

    outfile = datapath.parent / Path('corrupted_' + datapath.name)
    # Save the corrupted sequences if needed
    with open(outfile, "w") as f:
        for line in tqdm(corrupted_sequences, desc='Writing lines', unit=' lines'):
            f.write(line+'\n')
    #     f.write("\n".join(corrupted_sequences))


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset corruption augmenter')
    parser.add_argument('--datasets', type=str, required=True, nargs='+')
    # parser.add_argument('--out_dir', type=str)
    parser.add_argument('--cor_type', type=str, default='sub', choices='sub')
    parser.add_argument('--cor_rate', type=float, default=0.1)
    
    args = parser.parse_args()
    
    datapaths = [Path(patharg).resolve() for patharg in args.datasets]
    
    # out_path = Path(args.out_dir).resolve() 
    
    # print(datapaths[0].parent / Path('corrupted_' + datapaths[0].name))
    
    for p in datapaths:
        augment_dataset(p)
    