from pathlib import Path
import csv
import ast
import subprocess
import os
import time
import random
import soundfile as sf
import numpy as np
import yt_dlp
import resampy
from tqdm import tqdm
from collections import Counter


class StandardDownloader:
    def __init__(self,
                 data_file: str,
                 labels_file: str,
                 target_sr: int = 44100,
                 channels_proc: str = 'stereo',
                 normalize: bool = False,
                 cookies_file = None,
                 verbose: bool = False):
        """
        AudioSet standard dataset downloader with support for download tracking.

        :param data_file: Path to CSV containing video information.
        :param labels_file: Path to CSV containing label decoding information.
        :param target_sr: Target sampling rate (for audio resampling).
        :param channels_proc: Channel processing mode ('stereo', 'mono_split', or 'mono_red').
        :param normalize: Normalize audio to a peak amplitude of 1.0 if True.
        :param cookies_file: Path to cookies file for YouTibe user authentication.
        :param verbose: Enable debug logging if True.
        """
        self.data_file = Path(data_file)
        self.labels_file = Path(labels_file)
        self.target_sr = target_sr
        self.channels_proc = channels_proc
        self.normalize = normalize
        self.verbose = verbose
        self.cookies_file = cookies_file

        # Attributes for processing and reports tracking
        self.missing_samples = []
        self.downloaded_samples = []
        self.labels_counter = Counter()
        self.download_folder = self.create_output_folder()

        # yt-dlp options
        self.ydl_opts = {'quiet': not verbose,
                         'no_warnings': not verbose,
                         'format': 'bestaudio/best',
                         'outtmpl': '%(id)s.%(ext)s',
                         'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                         'sleep_requests': 1.25}
        if self.cookies_file:
            self.ydl_opts['cookiefile'] = self.cookies_file


    def _log(self, message: str):
        """Print messages if "verbose" mode is enabled."""
        if self.verbose:
            print(message)


    def __enter__(self):
        """Context manager entry point: load data and labels."""
        self.load_data()
        self.load_labels()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point: generate reports."""
        self.generate_reports()


    def load_data(self):
        """Load dataset from CSV file and check for 'downloaded' column."""
        if self.data_file.suffix != '.csv':
            raise TypeError("Unsupported dataset format. CSV required.")

        try:
            with self.data_file.open('r', newline='') as file:
                csv_reader = csv.DictReader(file)
                self.data = [row for row in csv_reader]

                # Ensure 'downloaded' column exists
                if 'downloaded' not in csv_reader.fieldnames:
                    for row in self.data:
                        row['downloaded'] = 'False'
                    self._save_data_to_csv()
                    self._log(f"Added 'downloaded' column to {self.data_file.name} with default 'False' values.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.data_file}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading the file: {str(e)}")


    def _save_data_to_csv(self):
        """Save the updated dataset back to the CSV file."""
        with self.data_file.open('w', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=self.data[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(self.data)


    def load_labels(self):
        """Load labels from CSV file."""
        try:
            with self.labels_file.open('r') as file:
                csv_reader = csv.DictReader(file)
                self.labels = {row['mid']: row['display_name'] for row in csv_reader}
            self._log(f'Loaded {len(self.labels)} labels from "{self.labels_file.name}".')
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file '{self.labels_file}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading the labels file: {str(e)}")


    def create_output_folder(self) -> Path:
        """Create a folder for downloads based on data filename."""
        folder_name = f"AudioSet_{self.data_file.stem}_downloads"
        download_folder = Path.cwd() / folder_name
        download_folder.mkdir(exist_ok=True)
        self._log(f"Downloads folder created: {download_folder}")
        return download_folder


    def download_and_process(self):
        """Download and process each audio sample with retry logic."""
        global_start_time = time.time()

        for idx, sample in enumerate(tqdm(self.data, desc="Dataset download & processing")):
            video_id = sample.get('yt_id')
            labels = ast.literal_eval(sample.get('positive_labels'))
            start_sec = float(sample.get('start_seconds'))
            end_sec = float(sample.get('end_seconds'))
            downloaded = sample.get('downloaded') == 'True'

            # Check if any .wav file containing the yt_id exists in the downloads folder
            audio_files = list(self.download_folder.glob(f"*{video_id}*.wav"))

            # Continue only if both the downloaded flag is True and audio files exist
            if downloaded and audio_files:
                self._log(f"Skipping already downloaded video ID: {video_id}.")
                continue

            if video_id and labels:
                label_names = [self.labels.get(label_id) for label_id in labels]
                self._log(f'Processing video ID: {video_id} with labels {label_names}.')
                while True:
                    try:
                        self.download_and_process_audio(video_id, label_names, start_sec, end_sec)
                        sample['downloaded'] = 'True'  # Mark as downloaded
                        break
                    except Exception as e:
                        error_message = str(e)
                        shadow_ban_messages = ["This content isn't available, try again later.",
                                               "Video unavailable. This content isn’t available.",
                                               "The following content is not available on this app.. Watch on the latest version of YouTube."]
                        if "Sign in to confirm you’re not a bot" in error_message:
                            sample['downloaded'] = 'False'
                            self._log(f"Authentication error for video ID '{video_id}'. Opening Firefox for manual cookie refresh.")
                            self.refresh_cookies()
                        elif any(msg in error_message for msg in shadow_ban_messages):
                            self._log(f"YouTube shadow-ban detected for video ID '{video_id}'. Entire downloading process halted.")
                            return  # Stop the entire downloading process
                        self._log(f"Error downloading video ID '{video_id}': {e}")
                        self.missing_samples.append({'video_id': video_id, 'labels': label_names})
                        break
            
            # Save the updated data back to the CSV file
            self._save_data_to_csv()

            # Add random sleep to relax connections
            sleep_time = random.uniform(5, 20)
            self._log(f"Sleeping for {sleep_time:.2f} seconds to avoid IP ban.")
            time.sleep(sleep_time)

        self._log(f"Dataset processed in {time.time() - global_start_time:.2f} seconds.")


    def refresh_cookies(self):
        """Open Firefox to allow manual cookie re-extraction."""
        subprocess.run(["firefox"], check=True)  # Launch Firefox
        self._log("Waiting for Firefox to close...")
        while "firefox" in subprocess.getoutput("pgrep firefox"):
            time.sleep(1)  # Wait until Firefox is closed
        self._log("Firefox closed. Reloading cookies.")
        if not self.cookies_file or not os.path.exists(self.cookies_file):
            raise FileNotFoundError("Cookies file not found. Please re-export the cookies file.")

        self.ydl_opts['cookiefile'] = self.cookies_file


    def download_and_process_audio(self, youtube_id: str, label_names: list, start_sec: float, end_sec: float):
        """Download and process a single audio sample."""
        try:
            self.ydl_opts['outtmpl'] = f'{youtube_id}.%(ext)s'
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([youtube_id])

            file_path = Path(f'{youtube_id}.wav')
            self.process_audio(file_path, start_sec, end_sec)
            self.downloaded_samples.append({'video_id': youtube_id, 'labels': label_names})
            self.labels_counter.update(label_names)
            self._log(f"Processed video ID: {youtube_id}")
        except Exception as e:
            raise e


    def process_audio(self, file_path: Path, start_sec: float, end_sec: float):
        """Apply DSP operations on audio file: resampling, trimming, normalization, and channel processing."""
        data, sr = sf.read(file_path)

        # Resampling
        if self.target_sr != sr:
            data = resampy.resample(data, sr, self.target_sr)
            self._log(f"{file_path} resampled to {self.target_sr}Hz.")

        # Trimming
        data = data[int(start_sec * self.target_sr): int(end_sec * self.target_sr)]

        # Normalization
        if self.normalize:
            data /= np.max(np.abs(data))
            self._log(f"{file_path} normalized to peak amplitude.")

        # Channels processing
        if self.channels_proc == 'mono_split' and data.ndim == 2 and data.shape[1] > 1:
            left_channel = data[:, 0]
            right_channel = data[:, 1]
            sf.write(f"{self.download_folder}/{file_path.stem}_Left.wav", left_channel, self.target_sr)
            sf.write(f"{self.download_folder}/{file_path.stem}_Right.wav", right_channel, self.target_sr)
        elif self.channels_proc == 'mono_red' and data.ndim == 2 and data.shape[1] > 1:
            data = (data[:, 0] + data[:, 1]) / 2.0
            sf.write(f"{self.download_folder}/{file_path.stem}_Reduced.wav", data, self.target_sr)
        else:
            sf.write(f"{self.download_folder}/{file_path.stem}_Original.wav", data, self.target_sr)

        # Remove the original downloaded file
        file_path.unlink()


    def generate_reports(self):
        """Generate reports for successful and failed downloads."""
        self._write_report("Missing_samples_report.txt", self.missing_samples)
        self._write_report("Success_samples_report.txt", self.downloaded_samples)
        self._write_report("Success_labels_report.txt", self.labels_counter.items(), count_report=True)


    def _write_report(self, filename: str, content, count_report=False):
        """Helper to write contents to TXT report files."""
        path = self.download_folder / filename
        with path.open("w") as file:
            for item in content:
                line = f"{item[0]}: {item[1]}\n" if count_report else f"{item['video_id']}: {item['labels']}\n"
                file.write(line)
        self._log(f"{filename} written.")
