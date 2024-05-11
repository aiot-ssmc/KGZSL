import tarfile

import libs.audio
import utils.file
from apps import N, SAMPLE_RATE, CHANNELS
from apps.data_process import SRC_MTG_DIR, MTG_DIR
from dl import x_dataset
from dl.data import InfoFrame

log = utils.log.get_logger()

artists_info = InfoFrame.load(MTG_DIR / 'metadata' / "artists_info.pkl")
albums_info = InfoFrame.load(MTG_DIR / 'metadata' / "albums_info.pkl")
genre_map = InfoFrame.load(MTG_DIR / 'label_map' / "genre.pkl")
instrument_map = InfoFrame.load(MTG_DIR / 'label_map' / "instrument.pkl")
mood_map = InfoFrame.load(MTG_DIR / 'label_map' / "mood_theme.pkl")

tag_start = 5

audio_load_func = libs.audio.load

x_wave_f = x_dataset.XWave("MP3", skip_encode=True)
x_features = {N.audio_data: x_wave_f, }
ffmpeg_encoder = x_wave_f.get_ffmpeg_encoder(quality="high")


def get_and_format_audio_data(path):
    dir_name, audio_name = path.split('/')
    tar_file = SRC_MTG_DIR / "audios" / f"raw_30s_audio-low-{dir_name}.tar"
    f = tarfile.open(tar_file, 'r').extractfile(f"{dir_name}/{audio_name.split('.')[0]}.low.mp3")
    audio_data = audio_load_func(f, channels=CHANNELS, frame_rate=SAMPLE_RATE)
    return audio_data


def check_x_data(audio_data, x_data):
    import io
    r_audio_data = audio_load_func(io.BytesIO(x_data), channels=CHANNELS, frame_rate=SAMPLE_RATE)
    import numpy
    diff = numpy.abs(audio_data - r_audio_data[:len(audio_data)])
    print(diff.mean() * 1000)


def iter_all_data():
    with (SRC_MTG_DIR / "autotagging.tsv").open('r') as reader:
        assert 'TAGS' == reader.readline().rstrip().split('\t')[tag_start]
        for i, line in enumerate(reader.readlines()):
            ls = line.rstrip().split('\t')
            track_id, artist_id, album_id, path, duration = ls[:tag_start]

            track_id, artist_id, album_id = map(lambda n: int(n.split('_')[-1]), (track_id, artist_id, album_id))

            duration = float(duration)
            audio_data = get_and_format_audio_data(path)
            x_data = ffmpeg_encoder(audio_data)

            # check_x_data(audio_data, x_data)

            genre_ids, instrument_ids, mood_ids = format_tags(ls[tag_start:])

            track_data = {
                N.track_id: track_id,
                N.album_id: album_id,
                N.artist_id: artist_id,

                N.audio_duration: duration,
                N.audio_data: x_data,

                N.genre_ids: genre_ids,
                N.instrument_ids: instrument_ids,
                N.mood_ids: mood_ids,
            }

            yield track_data


def format_tags(tags):
    genre_ids, instrument_ids, mood_ids = [], [], []
    for tag in tags:
        cls, tag_name = tag.split('---')
        if cls == 'genre':
            genre_ids.append(genre_map.name2id(tag_name))
        elif cls == 'instrument':
            instrument_ids.append(instrument_map.name2id(tag_name))
        elif cls == 'mood/theme':
            mood_ids.append(mood_map.name2id(tag_name))
        else:
            raise ValueError(f"unknown tag {tag}")

    return genre_ids, instrument_ids, mood_ids


def main():
    # for track_data in progressbar(iter_all_data()):
    #     a = track_data
    all_dataset = x_dataset.XDataset.from_generator(iter_all_data, x_features=x_features, num_proc=7)
    all_dataset.save_to_disk(MTG_DIR / f"mtg_all_data")
    print("done")


if __name__ == '__main__':
    main()
