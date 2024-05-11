import io
import zipfile

import libs.audio
import utils.file
from apps import N, SAMPLE_RATE, CHANNELS, MIN_DATA_ORIGINAL_LENGTH
from apps.data_process import FMA_DIR, SRC_FMA_DIR
from apps.data_process.fma import load_all_tracks_info
from dl import x_dataset
from dl.data import InfoFrame

log = utils.log.get_logger()

artists_info = InfoFrame.load(FMA_DIR / 'metadata' / "artists_info.pkl")
albums_info = InfoFrame.load(FMA_DIR / 'metadata' / "albums_info.pkl")
genre_map = InfoFrame.load(FMA_DIR / 'label_map' / "all_genres.pkl")

audio_load_func = libs.audio.load
check_valid = utils.data.isvalid

x_features = {N.audio_data: x_dataset.XWave("MP3"), }

archive_name = "fma_large"
audio_archive = zipfile.ZipFile(SRC_FMA_DIR / f"{archive_name}.zip", 'r')


def get_audio_filename(track_id):
    tid_str = '{:06d}'.format(track_id)
    return f"{archive_name}/{tid_str[:3]}/{tid_str}.mp3"


def load_track_audio(track_id):
    audio_file = get_audio_filename(track_id)
    audio_data = audio_archive.read(audio_file)
    try:
        audio_data = audio_load_func(io.BytesIO(audio_data), channels=CHANNELS, frame_rate=SAMPLE_RATE)
    except BrokenPipeError as e:
        log.warning(f"Audio file load failed: {audio_file}")
        log.warning(e)
        return None
    if len(audio_data) < MIN_DATA_ORIGINAL_LENGTH:
        log.warning(f"Audio file too short: {audio_file} ({len(audio_data) / SAMPLE_RATE}s)")
        return None
    return audio_data


def iter_all_data():
    all_tracks_info = load_all_tracks_info()
    for track_id, track_info in all_tracks_info.iterrows():
        album_id, artist_id = track_info['album']['id'], track_info['artist']['id']
        data_info = track_info['track'].to_dict()
        audio_data = load_track_audio(track_id)
        if audio_data is None:
            continue
        top_genre_id = genre_map.name2id(data_info['genre_top'])
        track_data = {
            N.track_id: track_id,
            N.album_id: album_id,
            N.artist_id: artist_id,

            # N.audio_duration: data_info['duration'],
            N.audio_data: audio_data,

            N.genre_ids: data_info['genres'],
            N.top_genre_id: top_genre_id,
            N.all_genre_ids: data_info['genres_all'],

            N.language: data_info['language_code'] if check_valid(data_info['language_code']) else '',
            N.name: data_info['title'] if check_valid(data_info['title']) else '',
            N.information: data_info['information'] if check_valid(data_info['information']) else '',
        }

        yield track_data


def main():
    # for track_data in iter_all_data():
    #     a = track_data
    all_dataset = x_dataset.XDataset.from_generator(iter_all_data, x_features=x_features)
    all_dataset.save_to_disk(FMA_DIR / f"fma_all_data")
    print("done")


if __name__ == '__main__':
    main()
