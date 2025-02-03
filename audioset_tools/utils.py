import csv
import json
import ast
from pathlib import Path
from typing import List
from collections import defaultdict


def find_samps_by_samps(targets_file: str, data_file: str, verbose: bool = False):
    """
    Find samples in data_file that match the yt_ids in targets_file.
    
    :param targets_file: File containing yt_ids to search for in data_file.
    :param data_file: File containing the data to search for matching yt_ids.
    :param verbose: Whether to print out the matching yt_ids.
    
    :returns tuple: 
        matching_rows: List(tuple(int, str))
            List of tuples containing the index of the matching row in data_file and the yt_id.
        num_matches: int, 
            The lenghth of the matching_rows list (amount of samples).
    
    Example:
    >>> matching_rows, size = find_samps_by_samps(targets_file='path/to/samples_you_are_searching_FOR.csv', 
                                                  data_file='path/to/samples_you_are_searching_IN.csv', 
                                                  verbose=True)
    """
    # Paths Handling
    cwd = Path.cwd()
    targets_file_path = cwd / targets_file
    data_file_path = cwd / data_file

    # Load yt_ids from targets_file into a set (fast lookup)
    yt_ids = set()
    with open(targets_file_path, 'r') as targets_file:
        reader = csv.DictReader(targets_file)
        for row in reader:
            yt_ids.add(row['yt_id'])
    
    # Find matching yt_ids in data_file and store metadata
    matching_rows = []
    with open(data_file_path, 'r') as data_file:
        reader = csv.DictReader(data_file)
        for index, row in enumerate(reader):
            if row['yt_id'] in yt_ids:
                matching_rows.append((index, row['yt_id']))
                if verbose:
                    print(f"Sample IDX: {index}, yt_id: {row['yt_id']}")

    return matching_rows, len(matching_rows)


def compute_stats(data_file: str, labels_file: str, verbose: bool = False, save_json: bool = False):
    """
    AudioSet (Processed) CSV statistics retriever: computes occurrences per label and total samples.
    Additionally, if the 'downloaded' attribute is present in the CSV header, the function computes:
      - The total number of samples marked as downloaded (True) and not downloaded (False).
      - Occurrences per label for downloaded and not downloaded samples, output in dictionary format:
        {'class': occurrence (int)}.
    
    Optionally, statistics are saved as a JSON file.
    
    :param data_file: Path to the CSV file containing video information.
    :param labels_file: Path to the CSV file containing labels decoding information.
    :param verbose: If True, enables debug printing. Default is False.
    :param save_json: If True, saves the computed statistics to a JSON file.
                      The JSON filename is generated as "<data_file>_stats.json".
    :return: A dictionary with the computed statistics. Example:
      {"total_samples": int,
       "label_occurrences": {<label>: count, ...},
       "downloaded_stats": {"downloaded": int,
                            "not_downloaded": int,
                            "label_occurrences_downloaded": {<label>: count, ...},
                            "label_occurrences_not_downloaded": {<label>: count, ...}}}
    
    Example:
    >>> stats = compute_stats(data_file='path/to/audioset_samples.csv',
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

    # Load label mapping
    with open(labels_file_path, 'r') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['mid']: row['display_name'] for row in csv_reader}
    if verbose:
        print(f"Loaded label mapping: {label_map}")

    # Initialize statistics dictionaries
    total_samples = 0
    label_occurrences = defaultdict(int)
    downloaded_count = 0
    not_downloaded_count = 0
    label_occ_downloaded = defaultdict(int)
    label_occ_not_downloaded = defaultdict(int)
    
    # Process dataset file
    with open(dataset_file_path, 'r') as dataset_file:
        reader = csv.reader(dataset_file)
        header = next(reader)
        
        # Determine if 'downloaded' attribute is present and its index (if any)
        try:
            downloaded_idx = header.index('downloaded')
            has_downloaded = True
            if verbose:
                print(f"'downloaded' attribute found at index {downloaded_idx}.")
        except ValueError:
            has_downloaded = False
            if verbose:
                print("No 'downloaded' attribute found in the provided CSV's header.")
        
        label_data_idx = 3  # default index for label data

        for row in reader:
            total_samples += 1
            try:
                labels_list = ast.literal_eval(row[label_data_idx])
                current_labels = []
                for item in labels_list:
                    current_labels.extend([lab.strip() for lab in item.split(',') if lab.strip()])
            except Exception as e:
                if verbose:
                    print(f"Error parsing labels in row {total_samples + 1}: {e}")
                current_labels = []

            decoded_labels = [label_map.get(label, label) for label in current_labels]

            # Update overall label occurrences (using decoded label names)
            for label in decoded_labels:
                label_occurrences[label] += 1

            # If downloaded attribute exists, update downloaded-related stats.
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

    # Prepare the result dictionary
    stats = {"total_samples": total_samples,
             "label_occurrences": dict(label_occurrences)}
    if has_downloaded:
        stats["downloaded_stats"] = {"downloaded": downloaded_count,
                                     "not_downloaded": not_downloaded_count,
                                     "label_occurrences_downloaded": dict(label_occ_downloaded),
                                     "label_occurrences_not_downloaded": dict(label_occ_not_downloaded)}
    if verbose:
        print("Statistics:")
        print(stats)
    
    # Save to JSON
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
    Merge multiple CSV files into a single one, removing duplicate rows.

    :param dataset_files: List of CSV files (paths) to merge.
    :param output_file: Path to the output file.
    :param verbose: Whether to print out the number of unique rows and the output file path.

    Example:
    >>> merged_CSV -> merge_sets(dataset_files=['path/to/set1.csv', 'path/to/set2.csv'], 
                                  output_file='path/to/output.csv')
    """
    unique_rows = set()  # Set to store unique rows

    with open(output_file, 'w', newline='') as fout:
        writer = None  # Init writer once headers are read

        # Main routine
        for filename in dataset_files:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Read headers

                if writer is None:
                    writer = csv.writer(fout)
                    writer.writerow(headers)  # Write headers once

                # Add each row to the set to ensure uniqueness
                for row in reader:
                    row_tuple = tuple(row)
                    if row_tuple not in unique_rows:
                        unique_rows.add(row_tuple)
                        writer.writerow(row)

    if verbose:
        print(f"Unique rows: {len(unique_rows)}")
        print(f"Output path: {output_file}")
