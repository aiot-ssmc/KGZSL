from collections import namedtuple

NAMES = [
    'track_id', 'album_id', 'artist_id', 'name',
    'genre_ids', 'instrument_ids', 'mood_ids',
    'top_genre_id', 'all_genre_ids',
    'audio_data', 'audio_duration',
    'language', 'information',

    'training', 'validation', 'testing',

    'feature', 'top_target', 'multi_targets',

    'ismir19_genre', 'top_genre',

    'seen', 'unseen', 'kg_embedding',

    'total_loss',
]
N = namedtuple('NAME', NAMES)(*NAMES)

CHANNELS = 1
SAMPLE_RATE = 16000
DATA_MIN_SECONDS = 1
MIN_DATA_LENGTH = DATA_MIN_SECONDS * SAMPLE_RATE

DATA_ORIGINAL_MIN_SECONDS = 5
MIN_DATA_ORIGINAL_LENGTH = DATA_ORIGINAL_MIN_SECONDS * SAMPLE_RATE
