"""AudioSet-Tools: taxonomy-aware curation of AudioSet, AudioSet Strong and VGGSound metadata.

Source files are parsed into one normalized record model (``AudioRecord``/``TemporalEvent``), manipulated through
``AudioDataset`` (label-aware filtering in the ``id``/``original``/``salt``/``auto`` query spaces, rebalancing,
merging, statistics, JSON/CSV export, event tracks) and, optionally, retrieved from YouTube with
``YouTubeDatasetDownloader``. Version 2 of the framework described in Giacomelli et al., EURASIP J. Audio Speech
Music Process. 2026 (doi 10.1186/s13636-025-00436-z).
"""

__version__ = "2.0.0"

from .dataset import AudioDataset, load_dataset, merge_datasets  # noqa: E402
from .downloader import (  # noqa: E402
    AUTH_REQUIRED_MESSAGES,
    SHADOW_BAN_MESSAGES,
    ConsecutiveFailuresError,
    DownloaderSetupError,
    DownloadHaltedError,
    ShadowBanError,
    YouTubeDatasetDownloader,
    read_download_report,
    write_download_report,
)
from .events import generate_event_tracks, iter_events, plot_events_pianoroll  # noqa: E402
from .filters import blacklist_labels, filter_labels, rebalance, select_range  # noqa: E402
from .models import (  # noqa: E402
    AUDIOSET,
    AUDIOSET_STRONG,
    VGGSOUND,
    AudioRecord,
    DownloadJob,
    DownloadReportEntry,
    TemporalEvent,
    is_valid_youtube_id,
)
from .parsers import detect_format, load_label_map, parse_metadata  # noqa: E402
from .salt import SaltAssetError, SaltResolver, default_salt_dir, fetch_salt_assets  # noqa: E402
from .utils import compute_stats, find_samples_by_samples, find_samples_by_youtube_ids, merge_files  # noqa: E402

__all__ = [
    "AUDIOSET",
    "AUDIOSET_STRONG",
    "AUTH_REQUIRED_MESSAGES",
    "SHADOW_BAN_MESSAGES",
    "VGGSOUND",
    "AudioDataset",
    "AudioRecord",
    "ConsecutiveFailuresError",
    "DownloadHaltedError",
    "DownloadJob",
    "DownloadReportEntry",
    "DownloaderSetupError",
    "SaltAssetError",
    "SaltResolver",
    "ShadowBanError",
    "TemporalEvent",
    "YouTubeDatasetDownloader",
    "__version__",
    "blacklist_labels",
    "compute_stats",
    "default_salt_dir",
    "detect_format",
    "fetch_salt_assets",
    "filter_labels",
    "find_samples_by_samples",
    "find_samples_by_youtube_ids",
    "generate_event_tracks",
    "is_valid_youtube_id",
    "iter_events",
    "load_dataset",
    "load_label_map",
    "merge_datasets",
    "merge_files",
    "parse_metadata",
    "plot_events_pianoroll",
    "read_download_report",
    "rebalance",
    "select_range",
    "write_download_report",
]
