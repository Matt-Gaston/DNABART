from pathlib import Path


def count_lines(filepath):
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)
    

def split_file_in_half(filepath):
    total_lines = count_lines(filepath)
    midpoint = total_lines // 2
    print(f'total lines: {total_lines}, half:{midpoint}')
    
    filepath = Path(filepath)
    part1 = filepath.parent / f"{filepath.stem}_part1{filepath.suffix}"
    part2 = filepath.parent / f"{filepath.stem}_part2{filepath.suffix}"
    
    with open(filepath, 'r') as infile, open(part1, 'w') as outfile1, open(part2, 'w') as outfile2:
        for i, line in enumerate(infile):
            if i < midpoint:
                outfile1.write(line)
            else:
                outfile2.write(line)
    
    print(f"File split into:\n  {part1}\n  {part2}")


split_file_in_half('../data/train.txt')
split_file_in_half('../data/corrupted_train.txt')