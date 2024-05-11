import pandas
import ast
from apps.data_process import SRC_FMA_DIR, FMA_DIR

SRC_FMA_META_DIR = SRC_FMA_DIR / "fma_metadata"


# extract from fma_metadata.zip


def _load_all_tracks():
    all_tracks_path = SRC_FMA_META_DIR / "tracks.csv"
    all_tracks_info = pandas.read_csv(all_tracks_path, index_col=0, header=[0, 1])

    columns = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all')]
    for column in columns:
        all_tracks_info[column] = all_tracks_info[column].map(ast.literal_eval)

    columns = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in columns:
        all_tracks_info[column] = pandas.to_datetime(all_tracks_info[column])

    subsets = ('small', 'medium', 'large')
    all_tracks_info['set', 'subset'] = all_tracks_info['set', 'subset'].astype(
        pandas.CategoricalDtype(categories=subsets, ordered=True))

    columns = [('track', 'genre_top'), ('track', 'license'),
               ('album', 'type'), ('album', 'information'),
               ('artist', 'bio')]
    for column in columns:
        all_tracks_info[column] = all_tracks_info[column].astype('category')

    return all_tracks_info


def load_all_tracks_info():
    gen_track_path = FMA_DIR / "tracks.pkl"
    if gen_track_path.exists():
        return pandas.read_pickle(gen_track_path)
    else:
        tracks = _load_all_tracks()
        tracks.to_pickle(gen_track_path)
        return tracks
