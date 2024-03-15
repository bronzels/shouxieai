import os
from tqdm import tqdm

def save_data(lines, file):
    with open(file, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, total=len(lines), desc=file):
            f.write('{}\n'.format(line))

def split_data(file):
    file_dir = os.path.dirname(file)
    train_file = os.path.join(file_dir, 'train_data.json')
    val_file = os.path.join(file_dir, 'dev_data.json')
    test_file = os.path.join(file_dir, 'test_data.json')
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    train_lines = lines[:len(lines) * 6 // 10]
    val_lines = lines[len(lines) * 6 // 10:len(lines) * 8 // 10]
    test_lines = lines[len(lines) * 8 // 10:]

    save_data(train_lines, train_file)
    save_data(val_lines, val_file)
    save_data(test_lines, test_file)

if __name__ == '__main__':
    split_data("all_data.json")

