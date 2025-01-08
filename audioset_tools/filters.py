import sys
from pathlib import Path
import ast
import csv
from typing import List, Optional
from collections import defaultdict, Counter
import random


def select_by_label(labels_file: str,
                    data_file: str,
                    target_labels: List[str],
                    out_filename: str,
                    verbose: bool = False):
    """
    AudioSet (Original) CSV segments filter: selection by ANY label(s) match.

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the CSV file containing video information.
    :param target_labels: List of searched labels in a human-readable format.
    :param out_filename: Path to the output CSV file for filtered data.
    :param verbose: If True, enables debug printing. Default is False.

    Example:
    >>> Speech_CSV -> select_by_label(labels_file='path/to/audioset_labels.csv', 
                                      data_file='path/to/audioset_samples.csv', 
                                      target_labels=["Speech"], 
                                      out_filename='path/to/output_file.csv', 
                                      verbose=True)
    """
    # Paths handling
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    # Ensure both files exist before proceeding
    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load labels mapping
    with open(labels_file_path, 'r') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet provided Labels-Map: {label_map}")

    # Get the target label IDs and convert them to a set (faster lookups)
    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels: {target_labels}", file=sys.stderr)
        print(f"Corresponding label IDs: {target_label_ids}", file=sys.stderr)

    # Filtering routine
    with open(dataset_file_path, 'r') as dataset_file, open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file)
        writer = csv.writer(output_file)

        # Write header and process rows
        for i, row in enumerate(reader):
            if i == 0:  # Write the header
                writer.writerow(row)
                continue

            # Re-format and clean positive labels
            clear_positive_labels = [label.strip().replace('"', '')
                                     for item in row[3:]
                                     for label in item.split(',')]

            # Write filtered rows based on label match
            if any(label in target_label_ids for label in clear_positive_labels):
                writer.writerow(row[:3] + [clear_positive_labels])

    if verbose:
        print(f"Filtered dataset CSV saved to {output_file_path}")


def blacklist_by_label(labels_file: str,
                       data_file: str,
                       target_labels: List[str],
                       out_filename: str,
                       verbose: bool = False):
    """
    AudioSet (Original) CSV segments filter: exclusion by ANY label(s) match.

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the CSV file containing video information.
    :param target_labels: List of labels in a human-readable format to exclude.
    :param out_filename: Path to the output CSV file for filtered data.
    :param verbose: If True, enables debug printing. Default is False.

    Example:
    >>> NO-Speech_CSV -> blacklist_by_label(labels_file='path/to/audioset_labels.csv', 
                                            data_file='path/to/audioset_samples.csv', 
                                            target_labels=["Speech"], 
                                            out_filename='path/to/output_file.csv', 
                                            verbose=True)
    """
    #Paths handling
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    # Ensure both files exist before proceeding
    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load labels mapping
    with open(labels_file_path, 'r') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet provided Labels-Map: {label_map}")

    # Get the target label IDs and convert them to a set (faster lookups)
    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels for exclusion: {target_labels}", file=sys.stderr)
        print(f"Corresponding label IDs for exclusion: {target_label_ids}", file=sys.stderr)

    # Filtering routine
    with open(dataset_file_path, 'r') as dataset_file, open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file)
        writer = csv.writer(output_file)

        # Write header and process rows
        for i, row in enumerate(reader):
            if i == 0:  # Write the header
                writer.writerow(row)
                continue

            # Re-format and clean positive labels
            clear_positive_labels = [label.strip().replace('"', '')
                                     for item in row[3:]
                                     for label in item.split(',')]

            # Write rows that DO NOT contain ANY target label
            if all(label not in target_label_ids for label in clear_positive_labels):
                writer.writerow(row[:3] + [clear_positive_labels])

    if verbose:
        print(f"Filtered dataset CSV (excluding specified labels) saved to {output_file_path}")


def select_by_samp_idx(data_file: str,
                       start_idx: int,
                       end_idx: int,
                       out_filename: str,
                       verbose: bool = False):
    """
    AudioSet CSV segments filter: selection by specific row interval (works w. both Original and Processed-CSVs)

    :param data_file: Path to the CSV file containing video information.
    :param start_idx: Starting row index for the selection (0-based).
    :param end_idxx: Ending row index for the selection (0-based, exclusive).
    :param out_filename: Path to the output CSV file for filtered data.
    :param verbose: If True, enables debug printing. Default is False.

    Example:
    >>> from_2_to_10_samples_CSV -> select_by_indices(data_file='path/to/audioset_samples.csv', 
                                                      start_index=2, 
                                                      end_index=10, 
                                                      out_filename='path/to/output_file.csv', 
                                                      verbose=True)
    """
    # Paths handling
    cwd = Path.cwd()
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    # Ensure the file exists before proceeding
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    if verbose:
        print(f"Selecting samples in [{start_idx}, {end_idx}[ interval.")

    # Filtering routine
    with open(dataset_file_path, 'r') as dataset_file, open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file)
        writer = csv.writer(output_file)

        # Write header and process rows
        for i, row in enumerate(reader):
            if i == 0:  # Write the header
                writer.writerow(row)
                continue

            # Write rows based on given interval
            if start_idx <= i < end_idx:
                writer.writerow(row)

    if verbose:
        print(f"Filtered dataset CSV saved to {output_file_path}")


def reselect_by_label(labels_file: str, 
                      data_file: str,
                      target_labels: List[str],
                      out_filename: str,
                      verbose: bool = False):
    """
    AudioSet CSV segments filter: selection by label(s) match (for PROCESSED CSVs)
    
    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the CSV file containing video information.
    :param target_labels: List of searched labels in a human-readable format.
    :param out_filename: Path to the output CSV file for filtered data.
    :param verbose: If True, enables debug printing. Default is False.

    Example:
    >>> Speech_CSV -> reselect_by_label(labels_file='path/to/audioset_labels.csv', 
                                        data_file='path/to/pre-processed/audioset_samples.csv', 
                                        target_labels=["Speech"], 
                                        out_filename='path/to/output_file.csv', 
                                        verbose=True)
    """
    cwd = Path.cwd()

    # Full paths retrieval
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    # Ensure both files exist before proceeding
    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")
    
    # Load label mapping
    with open(labels_file_path, 'r') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet provided Labels-Map: {label_map}")
    
    # Get the target label IDs and convert them to a set (faster lookups)
    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels: {target_labels}", file=sys.stderr)
        print(f"Corresponding label IDs: {target_label_ids}", file=sys.stderr)
    
    # Filtering routine
    with open(dataset_file_path, 'r') as dataset_file, open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file)
        writer = csv.writer(output_file)
        
        # Write header and process rows
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue

            # Re-format and clean positive labels
            clear_positive_labels = [label.strip().replace('"', '')
                                     for item in ast.literal_eval(row[3])
                                     for label in item.split(',')]
            
            # Write filtered rows based on ANY labels match
            if any(label in target_label_ids for label in clear_positive_labels):
                writer.writerow(row[:3] + [clear_positive_labels])
    
    if verbose:
        print(f"Filtered dataset CSV saved to {output_file_path}")


def reblacklist_by_label(labels_file: str,
                         data_file: str,
                         target_labels: List[str],
                         out_filename: str,
                         verbose: bool = False):
    """
    AudioSet CSV segments filter: exclusion by label(s) match (for PROCESSED CSVs)

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the CSV file containing video information.
    :param target_labels: List of labels in a human-readable format to exclude.
    :param out_filename: Path to the output CSV file for filtered data.
    :param verbose: If True, enables debug printing. Default is False.

    Example:
    >>> NO-Speech_CSV -> reblacklist_by_label(labels_file='path/to/audioset_labels.csv', 
                                              data_file='path/to/pre-processed/audioset_samples.csv', 
                                              target_labels=["Speech"], 
                                              out_filename='path/to/output_file.csv', 
                                              verbose=True)
    """
    # Paths handling
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    # Ensure both files exist before proceeding
    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load label mapping
    with open(labels_file_path, 'r') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet provided Labels-Map: {label_map}")

    # Get the target label IDs and convert them to a set for faster lookups
    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels for exclusion: {target_labels}", file=sys.stderr)
        print(f"Corresponding label IDs for exclusion: {target_label_ids}", file=sys.stderr)

    # Filtering routine
    with open(dataset_file_path, 'r') as dataset_file, open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file)
        writer = csv.writer(output_file)

        # Write header and process rows
        for i, row in enumerate(reader):
            if i == 0:  # Write the header
                writer.writerow(row)
                continue

            # Re-format and clean positive labels
            clear_positive_labels = [label.strip().replace('"', '')
                                     for item in ast.literal_eval(row[3])
                                     for label in item.split(',')]

            # Write rows that do not contain any target labels
            if all(label not in target_label_ids for label in clear_positive_labels):
                writer.writerow(row[:3] + [clear_positive_labels])

    if verbose:
        print(f"Filtered dataset CSV (excluding specified labels) saved to {output_file_path}")


def rebalancing_filter(input_csv: str,
                       labels_file: str,
                       output_csv: str,
                       focus_labels: Optional[List[str]] = None,
                       verbose: bool = False):
    """
    Advanced AudioSet CSV rebalancing filter, to handle samples with multiple labels (for PROCESSED CSVs)
    
    :param input_csv: Path to the input CSV file with sample metadata.
    :param labels_file: Path to the CSV file containing label decoding information.
    :param output_csv: Path to the output CSV file for the rebalanced dataset.
    :param focus_labels: List of human-readable labels to focus on for balancing.
                         If None, balances across all labels found in the input CSV.
    :param verbose: If True, enables debug printing. Default is False.

    Example:
    >>> Balanced_CSV -> rebalancing_filter(input_csv='path/to/input.csv',
                                           labels_file='path/to/audioset_labels.csv',
                                           output_csv='path/to/output.csv',
                                           focus_labels=['Car', 'Bus', 'Siren'],
                                           verbose=True)
    """
    input_path = Path(input_csv)
    labels_path = Path(labels_file)
    output_path = Path(output_csv)

    # Ensure the input and label files exist
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_csv} not found.")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")

    # Load label decoding map
    with open(labels_path, 'r') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}

    # Convert human-readable labels to encoded labels
    if focus_labels:
        focus_encoded_labels = {label_map[label] for label in focus_labels if label in label_map}
    else:
        # Use all labels if None specified
        focus_encoded_labels = set(label_map.values())  
    if verbose:
        print(f'Number of focus labels: {len(focus_encoded_labels)}')
        print(f"Focus labels (human-readable): {focus_labels}")
        print(f"Focus labels (encoded): {focus_encoded_labels}")
        
    # Load data and organize samples by labels
    with open(input_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    # Track samples for each label
    label_to_samples = defaultdict(list)
    for row in rows:
        positive_labels = set(ast.literal_eval(row[3]))
        common_labels = positive_labels.intersection(focus_encoded_labels)
        for label in common_labels:
            label_to_samples[label].append(row)

    # Determine target sample count (per label)
    min_count = min(len(samples) for samples in label_to_samples.values())
    if verbose:
        print(f"Target sample count per label: {min_count}")

    # Iterative balancing
    balanced_samples = set()
    used_samples = set()
    for label, samples in label_to_samples.items():
        random.shuffle(samples)
        selected = 0
        for sample in samples:
            sample_tuple = tuple(sample)
            if sample_tuple not in used_samples:
                balanced_samples.add(sample_tuple)
                used_samples.add(sample_tuple)
                selected += 1
            if selected >= min_count:
                break

    # Re-verify and adjust balancing
    balanced_samples = list(balanced_samples)
    final_counts = Counter(label for sample in balanced_samples
                           for label in set(ast.literal_eval(sample[3])).intersection(focus_encoded_labels))
    reverse_label_map = {v: k for k, v in label_map.items()}
    human_readable_counts = {reverse_label_map[label]: count for label, count in final_counts.items()}
    if verbose:
        print(f"Final label counts (human-readable): {human_readable_counts}")
        print(f"Total number of samples in the final CSV: {len(balanced_samples)}")

    # Write the rebalanced dataset to the output file
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(balanced_samples)
    if verbose:
        print(f"Rebalanced dataset saved to {output_csv}")
