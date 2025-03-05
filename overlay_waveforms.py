import argparse
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot waveforms of two WAV files with individual, overlay, and deviation plots.")
parser.add_argument("file_path1", type=str, help="Path to the first WAV file")
parser.add_argument("file_path2", type=str, help="Path to the second WAV file")
args = parser.parse_args()

# Extract subdirectory name to use in the output filename
subdir_name = os.path.basename(os.path.dirname(args.file_path1))

# Read the first wav file
sample_rate1, data1 = wavfile.read(args.file_path1)
time1 = np.linspace(0, len(data1) / sample_rate1, num=len(data1))

# Read the second wav file
sample_rate2, data2 = wavfile.read(args.file_path2)
time2 = np.linspace(0, len(data2) / sample_rate2, num=len(data2))

# Determine the maximum absolute amplitude across both waveforms
max_amplitude = max(np.max(np.abs(data1)), np.max(np.abs(data2)))

# Determine the maximum duration for x-axis alignment
max_duration = max(time1[-1], time2[-1])

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot the first waveform
axs[0].plot(time1, data1, label="First Audio Signal")
axs[0].set_ylim(-max_amplitude, max_amplitude)
axs[0].set_xlim(0, max_duration)
axs[0].set_ylabel("Amplitude")
axs[0].set_yticklabels([])

# Plot the second waveform
axs[1].plot(time2, data2, color="#ff7f7f", label="Second Audio Signal")
axs[1].set_ylim(-max_amplitude, max_amplitude)
axs[1].set_xlim(0, max_duration)
axs[1].set_ylabel("Amplitude")
axs[1].set_yticklabels([])
# Overlay plot with shaded deviation areas
axs[2].plot(time1, data1, alpha=0.6, label="Original Target Source")
axs[2].plot(time2, data2, color="#ff7f7f", alpha=0.6, label="Separated Source")
axs[2].fill_between(time1, data1, data2, where=(data1 > data2), facecolor='blue', alpha=0.2, interpolate=True)
axs[2].fill_between(time1, data1, data2, where=(data1 < data2), facecolor='red', alpha=0.2, interpolate=True)
axs[2].set_ylim(-max_amplitude, max_amplitude)
axs[2].set_xlim(0, max_duration)
axs[2].set_xlabel("Time (seconds)")
axs[2].set_ylabel("Amplitude")
axs[2].set_yticklabels([])
axs[2].legend()

plt.tight_layout()

# Save the plot as PNG with subdirectory-based filename in the current directory
output_filename = f"waveforms_combined_{subdir_name}.png"
output_path = os.path.join(os.getcwd(), output_filename)
plt.savefig(output_path, format="png")
print(f"Plot saved as {output_path}")

# Show the plot
#plt.show()
