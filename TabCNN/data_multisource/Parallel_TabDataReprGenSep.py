from TabDataReprGenSep import main
from multiprocessing import Pool
import sys
import argparse
import os
import shutil
# number of files to process overall
num_filenames = 360
modes = ["c","m","cm","s"]

filename_indices = list(range(num_filenames)) #* 4
mode_list = [modes[0]] * 360 #+ [modes[1]] * 360 + [modes[2]] * 360 + [modes[3]] * 360 


def preprocess_input_dir(input_dir):
    """
    Moves all subdirectories from 'train' and 'test' into input_dir
    and removes the empty 'train' and 'test' subdirectories.
    """
    subdirs = ["train", "test"]
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isdir(item_path):  # Check if it's a directory
                    shutil.move(item_path, input_dir)  # Move the subdirectory
            # Remove the now-empty directory
            os.rmdir(subdir_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create multi-source spectral representations out of waveforms.')

    # parser.add_argument('--input_dir', type=str, required=True, help="Input directory for processing files")
    parser.add_argument('--input_path', type=str, required=True, help="Input directory for processing files")
    args = parser.parse_args()

    preprocess_input_dir(args.input_path)

    # input_dirs = [args.input_path.split('/')[-1]] * len(filename_indices)
    input_dirs = [args.input_path] * len(filename_indices)
    # combined_args = zip(filename_indices, mode_list, itertools.repeat(args.input_dir))
    combined_args = zip(filename_indices, mode_list, input_dirs)
    # number of processes will run simultaneously
    pool = Pool(11)
    # results = pool.map(main, zip(filename_indices, mode_list)) #TODO include args.inputdir
    results = pool.map(main, combined_args)
