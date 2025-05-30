import sys
import os
audioset_tools_path = os.path.join(os.getcwd(), "audioset_tools")
sys.path.append(audioset_tools_path)

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from audioset_tools.filters_strong import select_by_label
from audioset_tools.utils_strong import plot_events_pianoroll, generate_event_tracks


# Create results directory
results_dir = "AudioSet_EV_strong"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
output_dir = results_dir + "/"

# Parameters #######################################################################
# Audioset CSV and TSV paths
audioset_csv_path = os.path.join(audioset_tools_path, "original_csv_01-11-2024/")
audioset_tsv_path = os.path.join(audioset_tools_path, "strong_tsv_12-03-2025/")

# Define the positives tuple: (list of target labels, output identifier)
positives = (['Emergency vehicle', 
              'Police car (siren)', 
              'Ambulance (siren)', 
              'Fire engine, fire truck (siren)'], 'EV_Positives')

# List of files to process and corresponding prefixes for output files.
files_to_process = [("audioset_train_strong.tsv", "train"),
                    ("audioset_eval_strong.tsv", "eval")]

# Process each file separately
for file_name, prefix in files_to_process:
    output_file = f"{positives[1]}_{prefix}.tsv"
    data_file_path = os.path.join(audioset_tsv_path, file_name)
    out_file_path = os.path.join(output_dir, output_file)
    
    # Filter by labels
    select_by_label(labels_file=os.path.join(audioset_csv_path, 'class_labels_indices.csv'),
                    data_file=data_file_path,
                    target_labels=positives[0],
                    out_filename=out_file_path,
                    verbose=False)
    
    # Count the number of samples in the resulting TSV (excluding header)
    output_path = Path(out_file_path)
    if output_path.exists():
        with open(output_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
            num_samples = len(rows) - 1  # subtract header
        print(f"Number of samples in {output_file}: {num_samples}")
    else:
        print(f"Output file {output_file} was not created.")

# -------------------------------------------------------------------------------
# Plot examples and additional assets (outside of the files processing loop)
# For these examples, we will use the filtered training file.
train_output_file = os.path.join(output_dir, positives[1] + "_train" + ".tsv")

# ----- Piano Roll Plotting Example -----
segment_id_to_plot = "6YORfRIpRr4_0"
figs = plot_events_pianoroll(data_file=train_output_file,
                             labels_file=os.path.join(audioset_csv_path, 'class_labels_indices.csv'),
                             segment_ids=segment_id_to_plot,
                             save_plots=True,
                             output_dir=output_dir,
                             verbose=False)

# ----- Torch Tensors Output Example -----
tracks = generate_event_tracks(data_file=train_output_file,
                               labels_file=os.path.join(audioset_csv_path, 'class_labels_indices.csv'),
                               bin_size=0.300,
                               out_type="pt",
                               verbose=False)

# ----- Plot Example of a Torch Track -----
if tracks:
    key, track = tracks[-1]
    # Convert torch.Tensor to numpy array if needed.
    track_np = track.numpy() if hasattr(track, "numpy") else track
    plt.figure(figsize=(10, 3))
    plt.title(f"Torch Track for {key}")
    plt.step(range(len(track_np)), track_np, where='post')
    plt.xlabel("Time Bin")
    plt.ylabel("Event Presence (1 or 0)")
    plt.grid(True)
    save_filename = "torch_track_example.png"
    plt.savefig(os.path.join(output_dir, save_filename), bbox_inches='tight')
