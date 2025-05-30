import os
import csv
import json
from pathlib import Path
from typing import Union, List
from collections import defaultdict
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


def find_samps_by_samps(targets_file: str, data_file: str, verbose: bool = False):
    """
    Find samples in data_file (AudioSet_Strong TSV) that match the yt_ids in targets_file (CSV).
    
    :param targets_file: File containing yt_ids to search for in data_file.
    :param data_file: TSV file containing the data to search for matching yt_ids.
    :param verbose: Whether to print out the matching yt_ids.
    
    :returns tuple: 
        matching_rows: List of tuples containing the index of the matching row in data_file and the yt_id.
        num_matches: int, the number of matching samples.
    
    Example:
    >>> matching_rows, size = find_samps_by_samps(targets_file='path/to/targets.csv', 
                                                  data_file='path/to/audioset_strong.tsv', 
                                                  verbose=True)
    """
    cwd = Path.cwd()
    targets_file_path = cwd / targets_file
    data_file_path = cwd / data_file

    # Load yt_ids from targets_file into a set for fast lookup.
    yt_ids = set()
    with open(targets_file_path, 'r') as tf:
        reader = csv.DictReader(tf)
        for row in reader:
            yt_ids.add(row['yt_id'])
    
    matching_rows = []
    # Open the AudioSet_Strong TSV with tab delimiters.
    with open(data_file_path, 'r', newline='') as df:
        reader = csv.DictReader(df, delimiter='\t')
        for index, row in enumerate(reader):
            # Extract yt_id from segment_id by splitting on underscore.
            segment_id = row['segment_id']
            video_id = segment_id.split('_')[0]
            if video_id in yt_ids:
                matching_rows.append((index, video_id))
                if verbose:
                    print(f"Sample IDX: {index}, yt_id: {video_id}")
    
    return matching_rows, len(matching_rows)


def compute_stats(data_file: str, labels_file: str, verbose: bool = False, save_json: bool = False):
    """
    AudioSet_Strong TSV statistics retriever: computes occurrences per label and total samples.
    Optionally, if a 'downloaded' attribute is present in the TSV header, computes additional stats.
    
    :param data_file: Path to the TSV file containing video information.
    :param labels_file: Path to the CSV file containing labels decoding information.
    :param verbose: If True, enables debug printing.
    :param save_json: If True, saves the computed statistics to a JSON file.
                      The JSON filename is generated as "<data_file>_stats.json".
    :return: A dictionary with computed statistics. Example:
      {
         "total_samples": int,
         "label_occurrences": {<label>: count, ...},
         "downloaded_stats": {
             "downloaded": int,
             "not_downloaded": int,
             "label_occurrences_downloaded": {<label>: count, ...},
             "label_occurrences_not_downloaded": {<label>: count, ...}
         }
      }
    
    Example:
    >>> stats = compute_stats(data_file='path/to/audioset_strong.tsv',
                              labels_file='path/to/audioset_labels.csv',
                              verbose=True,
                              save_json=True)
    """
    cwd = Path.cwd()
    dataset_file_path = cwd / data_file
    labels_file_path = cwd / labels_file

    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")
    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    
    # Load label mapping from CSV (comma-delimited)
    with open(labels_file_path, 'r', newline='') as lf:
        reader = csv.DictReader(lf)
        label_map = {row['mid']: row['display_name'] for row in reader}
    if verbose:
        print(f"Loaded label mapping: {label_map}")
    
    total_samples = 0
    label_occurrences = defaultdict(int)
    downloaded_count = 0
    not_downloaded_count = 0
    label_occ_downloaded = defaultdict(int)
    label_occ_not_downloaded = defaultdict(int)
    
    # Open the TSV file using tab delimiter.
    with open(dataset_file_path, 'r', newline='') as df:
        reader = csv.reader(df, delimiter='\t')
        header = next(reader)
        try:
            downloaded_idx = header.index('downloaded')
            has_downloaded = True
            if verbose:
                print(f"'downloaded' attribute found at index {downloaded_idx}.")
        except ValueError:
            has_downloaded = False
            if verbose:
                print("No 'downloaded' attribute found in the provided TSV header.")
        
        label_data_idx = 3  # In AudioSet_Strong, the label field is at index 3.
        
        for row in reader:
            total_samples += 1
            # In AudioSet_Strong, label is stored as a plain string.
            current_labels = [row[label_data_idx].strip()]
            decoded_labels = [label_map.get(label, label) for label in current_labels]
            for label in decoded_labels:
                label_occurrences[label] += 1
            
            if has_downloaded:
                downloaded_value = row[downloaded_idx].strip().lower()
                if downloaded_value in ['true', '1']:
                    downloaded_count += 1
                    for label in decoded_labels:
                        label_occ_downloaded[label] += 1
                else:
                    not_downloaded_count += 1
                    for label in decoded_labels:
                        label_occ_not_downloaded[label] += 1
    
    stats = {
        "total_samples": total_samples,
        "label_occurrences": dict(label_occurrences)
    }
    if has_downloaded:
        stats["downloaded_stats"] = {
            "downloaded": downloaded_count,
            "not_downloaded": not_downloaded_count,
            "label_occurrences_downloaded": dict(label_occ_downloaded),
            "label_occurrences_not_downloaded": dict(label_occ_not_downloaded)
        }
    
    if verbose:
        print("Statistics:")
        print(stats)
    
    if save_json:
        json_filename = f"{dataset_file_path.stem}_stats.json"
        json_path = dataset_file_path.parent / json_filename
        with open(json_path, 'w') as jf:
            json.dump(stats, jf, indent=4)
        if verbose:
            print(f"Statistics saved to {json_path}")
    
    return stats


def merge_sets(dataset_files: List[str], output_file: str, verbose: bool = False):
    """
    Merge multiple TSV files into a single one, removing duplicate rows.
    
    :param dataset_files: List of TSV file paths to merge.
    :param output_file: Path to the output file.
    :param verbose: Whether to print out the number of unique rows and the output file path.
    
    Example:
    >>> merge_sets(dataset_files=['path/to/set1.tsv', 'path/to/set2.tsv'], 
                   output_file='path/to/output.tsv', verbose=True)
    """
    unique_rows = set()
    cwd = Path.cwd()
    output_path = cwd / output_file
    with open(output_path, 'w', newline='') as fout:
        writer = None
        for filename in dataset_files:
            file_path = cwd / filename
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                headers = next(reader)
                if writer is None:
                    writer = csv.writer(fout, delimiter='\t')
                    writer.writerow(headers)
                for row in reader:
                    row_tuple = tuple(row)
                    if row_tuple not in unique_rows:
                        unique_rows.add(row_tuple)
                        writer.writerow(row)
    if verbose:
        print(f"Unique rows: {len(unique_rows)}")
        print(f"Output path: {output_path}")


def plot_events_pianoroll(data_file: str,
                          labels_file: str,
                          segment_ids: Union[str, List[str]],
                          save_plots: bool = False,
                          output_dir: str = None,
                          verbose: bool = False):
    """
    Parse the AudioSet_Strong TSV data_file and labels CSV file to produce a piano roll plot
    of events in the time domain. For each given segment_id (or list of them), the function:
    
      - Reads events (start_time_seconds, end_time_seconds, label) for that segment.
      - Uses the provided start_time_seconds and end_time_seconds directly (assuming the audio
        segment is 10 seconds long) to plot the events.
      - Plots each event as a horizontal colored bar in its corresponding label row.
      - Adds a legend mapping colors to labels.
      - Forces the x-axis to cover the full 10-second duration.
    
    :param data_file: Path to the AudioSet_Strong TSV file containing events.
                      Expected columns: segment_id, start_time_seconds, end_time_seconds, label.
    :param labels_file: Path to the CSV file containing labels mapping.
                        (Header: index, mid, display_name)
                        This function builds a mapping from mid to display_name.
    :param segment_ids: A single segment_id (str) or a list of segment_ids for which to generate plots.
    :param save_plots: If True, the generated plot(s) will be saved as PNG file(s).
    :param output_dir: Directory where to save plots (if save_plots is True). If None, uses current directory.
    :param verbose: If True, prints additional debug information.
    
    :return: A list of matplotlib Figure objects (one per segment_id). If a single segment_id is provided,
             the list will contain one figure.
             
    Example:
    >>> figs = plot_events_pianoroll(data_file='path/to/audioset_strong.tsv',
                                     labels_file='path/to/audioset_labels.csv',
                                     segment_ids=['s9d-2nhuJCQ_30000', 'abc123_45000'],
                                     save_plots=True,
                                     output_dir='plots',
                                     verbose=True)
    """
    # Ensure segment_ids is a list.
    if isinstance(segment_ids, str):
        segment_ids = [segment_ids]
    
    # Prepare file paths.
    data_path = Path(data_file)
    labels_path = Path(labels_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_file} not found.")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    
    if save_plots:
        if output_dir is None:
            output_dir = os.getcwd()
        else:
            os.makedirs(output_dir, exist_ok=True)
    
    # Load label mapping from CSV: mid -> display_name.
    label_mapping = {}
    with open(labels_path, 'r', newline='') as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            label_mapping[row['mid']] = row['display_name']
    if verbose:
        print("Loaded label mapping:")
        print(label_mapping)
    
    # Read the data file (TSV) and group events by segment_id.
    # Expected TSV header: segment_id, start_time_seconds, end_time_seconds, label
    events_by_segment = defaultdict(list)
    with open(data_path, 'r', newline='') as df:
        reader = csv.DictReader(df, delimiter='\t')
        for row in reader:
            seg_id = row['segment_id']
            # Only store events for the requested segment_ids.
            if seg_id in segment_ids:
                try:
                    # Instead of using any offset, use the provided times directly.
                    start_abs = float(row['start_time_seconds'])
                    end_abs = float(row['end_time_seconds'])
                except ValueError:
                    if verbose:
                        print(f"Skipping row with invalid times: {row}")
                    continue
                mid = row['label'].strip()
                # Map mid to human-readable label, if available.
                label = label_mapping.get(mid, mid)
                events_by_segment[seg_id].append({
                    'start': start_abs,
                    'end': end_abs,
                    'label': label
                })
    
    figures = []
    # For each segment, create a piano roll plot.
    for seg_id in segment_ids:
        events = events_by_segment.get(seg_id, [])
        if verbose:
            print(f"For segment {seg_id}, found {len(events)} event(s).")
        if not events:
            if verbose:
                print(f"No events found for segment {seg_id}. Skipping plot.")
            continue
        
        # Determine unique labels and assign a vertical position for each.
        unique_labels = sorted(set(event['label'] for event in events))
        label_to_y = {label: idx for idx, label in enumerate(unique_labels)}
        n_labels = len(unique_labels)
        
        # Set up colors using a colormap.
        cmap = cm.get_cmap('tab20', n_labels)
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
        
        fig, ax = plt.subplots(figsize=(10, 2 + n_labels))
        # Plot each event as a horizontal bar.
        for event in events:
            y = label_to_y[event['label']]
            start = event['start']
            duration = event['end'] - event['start']
            ax.broken_barh([(start, duration)], (y - 0.4, 0.8),
                           facecolors=label_to_color[event['label']])
        
        # Format the plot.
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Labels")
        ax.set_yticks(list(label_to_y.values()))
        ax.set_yticklabels(list(label_to_y.keys()))
        ax.set_title(f"Event Piano Roll for Segment {seg_id}")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        # Force the x-axis to span the full 10 seconds.
        ax.set_xlim(0, 10)
        
        # Create legend patches.
        legend_patches = [mpatches.Patch(color=label_to_color[label], label=label)
                          for label in unique_labels]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        figures.append(fig)
        
        if save_plots:
            out_filename = f"{seg_id}_pianoroll.png"
            out_path = Path(output_dir) / out_filename
            fig.savefig(out_path, bbox_inches='tight')
            if verbose:
                print(f"Saved plot for segment {seg_id} to {out_path}")
    
    return figures


def generate_event_tracks(data_file: str, 
                          labels_file: str, 
                          bin_size: float = 0.1, 
                          out_type: str = "np", 
                          verbose: bool = False):
    """
    Generate aggregated event tracks for AudioSet_Strong samples, creating one binary track per (segment_id, label)
    pair. The track is a binary vector for a fixed 10-second duration (the audio segment length) divided into bins
    of length bin_size. A bin is marked as 1 if any event for that (segment_id, label) overlaps that time interval,
    or 0 otherwise.
    
    In this version the relative times (start_time_seconds, end_time_seconds) are used directly without adding any
    offset from the segment_id.
    
    :param data_file: Path to the AudioSet_Strong TSV file with event annotations.
                      Expected columns: segment_id, start_time_seconds, end_time_seconds, label (MID).
    :param labels_file: Path to the CSV file with label mappings (columns: index, mid, display_name).
    :param bin_size: Time resolution (in seconds) for the event track. Default is 0.1 (i.e. 100ms bins).
    :param out_type: "np" to return NumPy ndarrays, "pt" to return torch.Tensors.
    :param verbose: If True, prints additional debug information.
    
    :return: A list of tuples. Each tuple is (segment_id_displayLabel, track), where track is a binary vector
             (NumPy array or torch.Tensor) of length L = int(10/bin_size).
             
    Example:
    >>> tracks = generate_event_tracks(data_file='audioset_strong.tsv',
                                       labels_file='audioset_labels.csv',
                                       bin_size=0.1,
                                       out_type="np",
                                       verbose=True)
    >>> for key, track in tracks:
    ...     print(key, track)
    """
    # Number of bins for a 10-second segment.
    L = int(10 / bin_size)
    
    # Load label mapping from CSV: mid -> display_name.
    label_mapping = {}
    labels_path = Path(labels_file)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    with open(labels_path, 'r', newline='') as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            label_mapping[row['mid']] = row['display_name']
    if verbose:
        print("Loaded label mapping:")
        for mid, disp in label_mapping.items():
            print(f"  {mid} -> {disp}")
    
    # Dictionary to accumulate aggregated tracks per (segment_id, label).
    aggregated_tracks = {}
    
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_file} not found.")
    
    with open(data_path, 'r', newline='') as df:
        reader = csv.DictReader(df, delimiter='\t')
        for row in reader:
            seg_id = row['segment_id']
            try:
                rel_start = float(row['start_time_seconds'])
                rel_end   = float(row['end_time_seconds'])
            except ValueError:
                if verbose:
                    print(f"Skipping row with invalid times: {row}")
                continue
            
            # Use the provided relative times directly (clip them to [0,10]).
            event_start = max(0.0, min(rel_start, 10.0))
            event_end   = max(0.0, min(rel_end, 10.0))
            
            # Convert event times to bin indices.
            start_idx = int(event_start / bin_size)
            end_idx = int(math.ceil(event_end / bin_size))
            if end_idx > L:
                end_idx = L
            
            # Create a binary track for this event.
            event_track = np.zeros(L, dtype=int)
            event_track[start_idx:end_idx] = 1
            
            # Convert MID to human-readable label.
            mid = row['label'].strip()
            display_label = label_mapping.get(mid, mid)
            
            # Build the aggregated key: "segment_id_displayLabel".
            key = f"{seg_id}_{display_label}"
            
            if key in aggregated_tracks:
                # Aggregate events using element-wise OR (logical maximum).
                aggregated_tracks[key] = np.maximum(aggregated_tracks[key], event_track)
            else:
                aggregated_tracks[key] = event_track.copy()
            
            if verbose:
                print(f"Segment {seg_id}: event [{event_start:.2f}, {event_end:.2f}] for label '{display_label}' -> key '{key}' (bins {start_idx} to {end_idx})")
    
    # Convert aggregated tracks to the desired output type.
    output_tracks = []
    for key, track in aggregated_tracks.items():
        if out_type == "pt":
            if torch is None:
                raise ImportError("Torch is not installed; cannot output torch.Tensor.")
            track_out = torch.from_numpy(track)
        else:
            track_out = track
        output_tracks.append((key, track_out))
    
    return output_tracks
