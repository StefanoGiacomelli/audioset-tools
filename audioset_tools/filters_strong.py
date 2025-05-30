import sys
from pathlib import Path
import csv
from typing import List, Optional
from collections import defaultdict, Counter
import random


# Original AudioSet_Strong TSV functions
def select_by_label(labels_file: str,
                    data_file: str,
                    target_labels: List[str],
                    out_filename: str,
                    verbose: bool = False):
    """
    AudioSet_Strong TSV segments filter: selection by ANY label(s) match.

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the TSV file containing video information.
    :param target_labels: List of searched labels in human-readable format.
    :param out_filename: Path to the output TSV file for filtered data.
    :param verbose: If True, enables debug printing.
    """
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load labels mapping from CSV (comma-delimited)
    with open(labels_file_path, 'r', newline='') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet_Strong Labels Map: {label_map}")

    # Get target label IDs
    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels: {target_labels}")
        print(f"Corresponding label IDs: {target_label_ids}")

    # Filtering routine using TSV delimiters for the data file
    with open(dataset_file_path, 'r', newline='') as dataset_file, \
         open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        for i, row in enumerate(reader):
            if i == 0:  # header
                writer.writerow(row)
                continue

            # For AudioSet_Strong the label is in the 4th column; wrap it in a list.
            clear_positive_labels = [row[3].strip()]
            if any(label in target_label_ids for label in clear_positive_labels):
                writer.writerow(row)
    if verbose:
        print(f"Filtered dataset TSV saved to {output_file_path}")


def blacklist_by_label(labels_file: str,
                       data_file: str,
                       target_labels: List[str],
                       out_filename: str,
                       verbose: bool = False):
    """
    AudioSet_Strong TSV segments filter: exclusion by ANY label(s) match.

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the TSV file containing video information.
    :param target_labels: List of labels in human-readable format to exclude.
    :param out_filename: Path to the output TSV file for filtered data.
    :param verbose: If True, enables debug printing.
    """
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load labels mapping from CSV (comma-delimited)
    with open(labels_file_path, 'r', newline='') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet_Strong Labels Map: {label_map}")

    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels for exclusion: {target_labels}")
        print(f"Corresponding label IDs for exclusion: {target_label_ids}")

    # Filtering routine using TSV delimiters for the data file
    with open(dataset_file_path, 'r', newline='') as dataset_file, \
         open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        for i, row in enumerate(reader):
            if i == 0:  # header
                writer.writerow(row)
                continue

            clear_positive_labels = [row[3].strip()]
            if all(label not in target_label_ids for label in clear_positive_labels):
                writer.writerow(row)
    if verbose:
        print(f"Filtered dataset TSV (excluding specified labels) saved to {output_file_path}")


def select_by_samp_idx(data_file: str,
                       start_idx: int,
                       end_idx: int,
                       out_filename: str,
                       verbose: bool = False):
    """
    AudioSet_Strong TSV segments filter: selection by specific row interval.

    :param data_file: Path to the TSV file containing video information.
    :param start_idx: Starting row index for the selection (0-based).
    :param end_idx: Ending row index for the selection (0-based, exclusive).
    :param out_filename: Path to the output TSV file for filtered data.
    :param verbose: If True, enables debug printing.
    """
    cwd = Path.cwd()
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    if verbose:
        print(f"Selecting samples in the interval [{start_idx}, {end_idx}[.")

    with open(dataset_file_path, 'r', newline='') as dataset_file, \
         open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        for i, row in enumerate(reader):
            if i == 0:  # header
                writer.writerow(row)
                continue
            if start_idx <= i < end_idx:
                writer.writerow(row)
    if verbose:
        print(f"Filtered dataset TSV saved to {output_file_path}")


# Processed AudioSet Strong TSV functions
def reselect_by_label(labels_file: str,
                      data_file: str,
                      target_labels: List[str],
                      out_filename: str,
                      verbose: bool = False):
    """
    AudioSet_Strong TSV segments filter (processed version): selection by label(s) match.

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the TSV file containing video information.
    :param target_labels: List of searched labels in human-readable format.
    :param out_filename: Path to the output TSV file for filtered data.
    :param verbose: If True, enables debug printing.
    """
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load labels mapping from CSV (comma-delimited)
    with open(labels_file_path, 'r', newline='') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet_Strong Labels Map: {label_map}")

    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels: {target_labels}")
        print(f"Corresponding label IDs: {target_label_ids}")

    with open(dataset_file_path, 'r', newline='') as dataset_file, \
         open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue

            clear_positive_labels = [row[3].strip()]
            if any(label in target_label_ids for label in clear_positive_labels):
                writer.writerow(row[:3] + [str(clear_positive_labels)])
    if verbose:
        print(f"Filtered processed dataset TSV saved to {output_file_path}")


def reblacklist_by_label(labels_file: str,
                         data_file: str,
                         target_labels: List[str],
                         out_filename: str,
                         verbose: bool = False):
    """
    AudioSet_Strong TSV segments filter (processed version): exclusion by label(s) match.

    :param labels_file: Path to the CSV file containing labels decoding information.
    :param data_file: Path to the TSV file containing video information.
    :param target_labels: List of labels in human-readable format to exclude.
    :param out_filename: Path to the output TSV file for filtered data.
    :param verbose: If True, enables debug printing.
    """
    cwd = Path.cwd()
    labels_file_path = cwd / labels_file
    dataset_file_path = cwd / data_file
    output_file_path = cwd / out_filename

    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    # Load labels mapping from CSV (comma-delimited)
    with open(labels_file_path, 'r', newline='') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}
        if verbose:
            print(f"AudioSet_Strong Labels Map: {label_map}")

    target_label_ids = {label_map[label] for label in target_labels if label in label_map}
    if not target_label_ids:
        print("No valid target labels found in the label map.", file=sys.stderr)
        return None
    if verbose:
        print(f"Target labels for exclusion: {target_labels}")
        print(f"Corresponding label IDs for exclusion: {target_label_ids}")

    with open(dataset_file_path, 'r', newline='') as dataset_file, \
         open(output_file_path, 'w', newline='') as output_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        writer = csv.writer(output_file, delimiter='\t')

        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
                continue

            clear_positive_labels = [row[3].strip()]
            if all(label not in target_label_ids for label in clear_positive_labels):
                writer.writerow(row[:3] + [str(clear_positive_labels)])
    if verbose:
        print(f"Filtered processed dataset TSV (excluding specified labels) saved to {output_file_path}")


def rebalancing_filter(input_tsv: str,
                       labels_file: str,
                       output_tsv: str,
                       focus_labels: Optional[List[str]] = None,
                       verbose: bool = False):
    """
    Advanced AudioSet_Strong TSV rebalancing filter to handle samples with single labels (processed version).

    :param input_tsv: Path to the input TSV file with sample metadata.
    :param labels_file: Path to the CSV file containing labels decoding information.
    :param output_tsv: Path to the output TSV file for the rebalanced dataset.
    :param focus_labels: List of human-readable labels to focus on for balancing.
                         If None, balances across all labels found in the input TSV.
    :param verbose: If True, enables debug printing.
    """
    input_path = Path(input_tsv)
    labels_path = Path(labels_file)
    output_path = Path(output_tsv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_tsv} not found.")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file {labels_file} not found.")

    # Load label decoding map from CSV (comma-delimited)
    with open(labels_path, 'r', newline='') as lf:
        csv_reader = csv.DictReader(lf)
        label_map = {row['display_name']: row['mid'] for row in csv_reader}

    if focus_labels:
        focus_encoded_labels = {label_map[label] for label in focus_labels if label in label_map}
    else:
        focus_encoded_labels = set(label_map.values())
    if verbose:
        print(f'Number of focus labels: {len(focus_encoded_labels)}')
        print(f"Focus labels (human-readable): {focus_labels}")
        print(f"Focus labels (encoded): {focus_encoded_labels}")

    # Load data from TSV
    with open(input_path, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter='\t')
        header = next(reader)
        rows = list(reader)

    # Organize samples by labels
    label_to_samples = defaultdict(list)
    for row in rows:
        # In AudioSet_Strong, there is a single label; wrap it in a set.
        positive_labels = {row[3].strip()}
        common_labels = positive_labels.intersection(focus_encoded_labels)
        for label in common_labels:
            label_to_samples[label].append(row)

    if not label_to_samples:
        print("No samples found for the focus labels.", file=sys.stderr)
        return None

    min_count = min(len(samples) for samples in label_to_samples.values())
    if verbose:
        print(f"Target sample count per label: {min_count}")

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

    balanced_samples = list(balanced_samples)
    final_counts = Counter(label for sample in balanced_samples
                           for label in {sample[3].strip()}.intersection(focus_encoded_labels))
    reverse_label_map = {v: k for k, v in label_map.items()}
    human_readable_counts = {reverse_label_map[label]: count for label, count in final_counts.items()}
    if verbose:
        print(f"Final label counts (human-readable): {human_readable_counts}")
        print(f"Total number of samples in the final TSV: {len(balanced_samples)}")

    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(header)
        writer.writerows(balanced_samples)
    if verbose:
        print(f"Rebalanced dataset TSV saved to {output_tsv}")
