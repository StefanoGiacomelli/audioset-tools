import os
import sys
audioset_tools_path = os.path.join(os.getcwd(), "audioset_tools")
sys.path.append(audioset_tools_path)
from audioset_tools.downloaders import StandardDownloader


class_labels_path = './audioset_tools/original_csv_01-11-2024/class_labels_indices.csv'


# Positive samples downloader
with StandardDownloader(data_file='./AudioSet_EV/EV_Positives.csv',
                        labels_file=class_labels_path,
                        target_sr=32000,
                        channels_proc='mono_red',
                        normalize=False,
                        cookies_file='cookies.txt',
                        verbose=False) as downloader:
    downloader.download_and_process()

# Negative samples downloader
with StandardDownloader(data_file='./AudioSet_EV/EV_Negatives.csv',
                        labels_file=class_labels_path,
                        target_sr=32000,
                        channels_proc='mono_red',
                        normalize=False,
                        cookies_file='cookies.txt',
                        verbose=False) as downloader:
    downloader.download_and_process()
