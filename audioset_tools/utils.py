import csv
from pathlib import Path
from typing import List


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
