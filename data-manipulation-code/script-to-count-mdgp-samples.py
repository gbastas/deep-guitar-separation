import os
import numpy as np

# Path to the directory containing guitar samples
base_path = "note_instances_mic/data/train"  # Replace with the path to your dataset

# Initialize the array to hold counts (24 frets x 6 strings)
note_count = np.zeros((24, 6), dtype=int)  # 24 frets and 6 strings

# Define the strings and frets
strings = [1,2, 3, 4, 5, 6]  # Strings 3 to 6
frets = range(0, 25)  # Frets 10 to 24

# Iterate over the guitars (guitar96, guitar97, etc.)
for guitar_dir in os.listdir(base_path):
    guitar_path = os.path.join(base_path, guitar_dir)
    
    if os.path.isdir(guitar_path):  # Check if it's a directory
        for string in strings:
            string_dir = f"string{string}"
            string_path = os.path.join(guitar_path, string_dir)
            
            if os.path.isdir(string_path):  # Check if the string directory exists
                for fret in frets:
                    fret_file = f"{fret}.wav"
                    fret_file_path = os.path.join(string_path, fret_file)
                    
                    if os.path.exists(fret_file_path):  # Check if the .wav file exists
                        fret_index = fret   # Index for the 24 frets (10-24 -> 0-14)
                        string_index = string - 1  # Index for the 6 strings (3-6 -> 0-3)
                        
                        # Increment the note count for this string and fret
                        note_count[fret_index, string_index] += 1

# Print the result
print("Note count (24 frets x 6 strings):")
print(note_count)

# How it Works:

#     Directory Traversal: The script traverses the directories for each guitar (guitar96, guitar97, etc.) and within those, it looks for subdirectories corresponding to the strings (string3, string4, etc.).

#     File Counting: For each string and fret combination (e.g., string3 and fret10), the script checks if a .wav file exists and increments the count for that string-fret pair in the note_count array.

#     Array Structure: The resulting note_count array is of size 24x6 (representing 24 frets and 6 strings). Each element in the array will contain the count of samples available for the corresponding fret and string.

# Output Example:

# If the directory structure contains samples for guitar96, guitar97, and guitar98, and each has recordings for frets 10–12 on each string, the printed output might look like:

# Note count (24 frets x 6 strings):
# [[0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [1 0 0 0 1 0]
#  [1 0 0 0 1 0]
#  [1 0 0 0 1 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]
#  [0 0 0 0 0 0]]

# Visualization Example (Optional):

# If you would like to visualize the result as a heatmap, you can use matplotlib to create a color-coded plot of the counts:

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 8))
# plt.imshow(note_count, cmap='Blues', aspect='auto', origin='lower')
# plt.colorbar(label="Number of samples")
# plt.xlabel('String')
# plt.ylabel('Fret')
# plt.title('Number of Samples per Fret-String Combination')
# plt.xticks(np.arange(6), ['String 3', 'String 4', 'String 5', 'String 6'])
# plt.yticks(np.arange(24), [str(i) for i in range(10, 25)])

# plt.show()