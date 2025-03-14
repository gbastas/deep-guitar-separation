import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir_to_split', type=str, default='pseudo_sep_all_solos',
                    help="List of instruments to separate (default: \"bass drums other vocals\")") # __gbastas__

args = parser.parse_args()


pseudo_sep_path = args.dir_to_split 
datasep_mic_test_path = "../datasets/GuitarSet/datasep-mic/test/"
os.makedirs(pseudo_sep_path+'/test/', exist_ok=True)
for dir_name in os.listdir(pseudo_sep_path): # e.g. 01_BN3-119-G_comp_hex_mic
    source_dir = os.path.join(pseudo_sep_path, dir_name)
    dest_dir_name = '_'.join(dir_name.split('_')[:3]) # e.g. 01_BN3-119-G_comp
    dest_dir = os.path.join(datasep_mic_test_path, dest_dir_name)


    # print('os.path.exists(pseudo_sep_path):', os.path.exists(pseudo_sep_path))
    if not os.path.exists(pseudo_sep_path):
        raise FileNotFoundError(f"Directory {pseudo_sep_path} does not exist.")

    # print('os.path.exists(source_dir):', os.path.isdir(source_dir))
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Expected a directory but found: {source_dir}")

    if os.path.isdir(source_dir) and os.path.exists(dest_dir):
        # print(dest_dir)
        # print(source_dir)
        shutil.move(source_dir, os.path.join(pseudo_sep_path, "test"))


datasep_mic_train_path = "../datasets/GuitarSet/datasep-mic/train/"
os.makedirs(pseudo_sep_path + '/train/', exist_ok=True)

for dir_name in os.listdir(pseudo_sep_path):  # e.g. 01_BN3-119-G_comp_hex_mic
    source_dir = os.path.join(pseudo_sep_path, dir_name)
    dest_dir_name = '_'.join(dir_name.split('_')[:3])  # e.g. 01_BN3-119-G_comp
    dest_dir = os.path.join(datasep_mic_train_path, dest_dir_name)

    if not os.path.exists(pseudo_sep_path):
        raise FileNotFoundError(f"Directory {pseudo_sep_path} does not exist.")

    if not os.path.isdir(source_dir):
        continue  # Skip if it's not a directory

    if os.path.isdir(source_dir) and os.path.exists(dest_dir):
        # print(dest_dir)
        # print(source_dir)
        shutil.move(source_dir, os.path.join(pseudo_sep_path, "train"))