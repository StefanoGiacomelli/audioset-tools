import sys
import os
from tqdm import tqdm
audioset_tools_path = os.path.join(os.getcwd(), "audioset_tools")
sys.path.append(audioset_tools_path)
import csv
import random
from audioset_tools.filters import select_by_label, reblacklist_by_label, rebalancing_filter
from audioset_tools.utils import merge_sets


SEED = 42
random.seed(SEED)
verbose = False

# Parameters #######################################################################
# Audioset CSV segments paths
audioset_csv_path = os.path.join(audioset_tools_path, "original_csv_01-11-2024/")
audioset_csv_filespath = [os.path.join(audioset_csv_path, f) 
                          for f in os.listdir(audioset_csv_path) if f.endswith('segments.csv')]

# Build Positive (label=1) and Negative (label=0) groups
positives = (['Emergency vehicle',                  # CONTAINER
              'Police car (siren)',
              'Ambulance (siren)',
              'Fire engine, fire truck (siren)'], 'EV_Positives')

# Old Positives (statistically relevant) labels included --> up to 10^-3 (in out_probs)
negatives = (['Car',                                # Vehicles Sounds (CONTAINER) ---- 
              'Car passing by', 
              'Power windows, electric windows',
              'Tire squeal',
              'Motor vehicle (road)',  				# CONTAINER
              'Truck',  							# CONTAINER
              'Air brake',
              'Ice cream truck, ice cream van',
              'Bus', 
              'Motorcycle',
              'Skidding',
              'Race car, auto racing',
              'Bicycle', 							# CONTAINER
              'Train',								# CONTAINER
              'Rail transport', 					# CONTAINER
              'Train wheels squealing',
              'Railroad car, train wagon',
              'Skateboard',
              'Traffic noise, roadway noise',       # Environments -------------------																			# Environments
              'Outside, rural or natural', 
              'Outside, urban or manmade',
              'Car alarm',                          # Alarms -------------------------                 																					# Alarms           
              'Vehicle horn, car horn, honking',	# CONTAINER
              'Bicycle bell',
              'Train horn',
              'Train whistle',
              'Foghorn',
              'Toot',
              'Reversing beeps',
              'Beep, bleep',
              'Civil defense siren',
              'Alarm',								# CONTAINER
              'Smoke detector, smoke alarm',
              'Fire alarm',
              'Buzzer',
              'Speech',                             # Others (CONTAINERS) ------------                        																					# Others (statistically relevant)
              'Music',
              'Singing',
              'Engine'], 'EV_Negatives')


# POSITIVE samples pipeline ########################################################
# 1) Select positive samples from all AudioSet segments
processed_file_paths = []
for segment_path in audioset_csv_filespath:
    segment_name = os.path.basename(segment_path).split('.')[0][:-9]
    select_by_label(labels_file=audioset_csv_path + 'class_labels_indices.csv',
                    data_file=segment_path,
                    target_labels=positives[0],
                    out_filename=f'./{segment_name}_{positives[1]}.csv',
                    verbose=verbose)
    processed_file_paths.append(f'./{segment_name}_{positives[1]}.csv')

# 2) Merge all positive CSV files & delete pre-processed files
merge_sets(dataset_files=processed_file_paths,
           output_file=f'./{positives[1]}_non-blacklisted.csv',
           verbose=verbose)
for file in processed_file_paths:
    os.remove(file)

# 3) Blacklist "Civil defense siren" & delete non-blacklisted CSV file
reblacklist_by_label(labels_file=audioset_csv_path + 'class_labels_indices.csv',
                     data_file=f'./{positives[1]}_non-blacklisted.csv',
                     target_labels=['Civil defense siren'],
                     out_filename=f'./{positives[1]}.csv',
                     verbose=verbose)
os.remove(f'./{positives[1]}_non-blacklisted.csv')


####################################################################################
# NEGATIVE samples pipeline:
# 1) Select negative samples from all AudioSet segments
processed_file_paths = []
for segment_path in audioset_csv_filespath:
    segment_name = os.path.basename(segment_path).split('.')[0][:-9]
    select_by_label(labels_file=audioset_csv_path + 'class_labels_indices.csv',
                    data_file=segment_path,
                    target_labels=negatives[0],
                    out_filename=f'./{segment_name}_{negatives[1]}.csv',
                    verbose=verbose)
    processed_file_paths.append(f'./{segment_name}_{negatives[1]}.csv')

# 2) Merge all negative CSV files & delete pre-processed files
merge_sets(dataset_files=processed_file_paths,
           output_file=f'./{negatives[1]}_non-blacklisted.csv',
           verbose=verbose)
for file in processed_file_paths:
    os.remove(file)

# 3) Blacklist all positive labels & delete non-blacklisted CSV file
reblacklist_by_label(labels_file=audioset_csv_path + 'class_labels_indices.csv',
                     data_file=f'./{negatives[1]}_non-blacklisted.csv',
                     target_labels=positives[0],
                     out_filename=f'./{negatives[1]}_blacklisted.csv',
                     verbose=verbose)
os.remove(f'./{negatives[1]}_non-blacklisted.csv')

# 4) Blacklisted Negative samples class rebalancing
rebalancing_filter(input_csv=f'./{negatives[1]}_blacklisted.csv',
                   labels_file=audioset_csv_path + 'class_labels_indices.csv',
                   output_csv=f'./{negatives[1]}.csv',
                   focus_labels=negatives[0],
                   verbose=True)
os.remove(f'./{negatives[1]}_blacklisted.csv')


####################################################################################
# Count samples per group
positives_csv = f'./{positives[1]}.csv'
negatives_csv = f'./{negatives[1]}.csv'
with open(positives_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    positives_count = sum(1 for row in reader)
with open(negatives_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    negatives_count = sum(1 for row in reader)

print(f'Positive samples: ', positives_count)
print(f'Negative samples: ', negatives_count)
# Note: check column headers formattting in CSV files: no spaces after commas!
