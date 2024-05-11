import pandas

import utils.file
from apps.data_process import FMA_DIR
from apps.data_process.fma import load_all_tracks_info
from dl.data import InfoFrame
from utils.log import progressbar

log = utils.log.get_logger()


def get_artists_info(all_tracks_info: pandas.DataFrame) -> InfoFrame:
    all_artists = {}
    for track_id, track_info in progressbar(all_tracks_info.iterrows(), length=len(all_tracks_info)):
        artist_info = track_info['artist'].to_dict()
        artist_id = artist_info.pop('id')
        if artist_id not in all_artists:
            all_artists[artist_id] = artist_info
        else:
            assert utils.none_equal(all_artists[artist_id]['name'], artist_info['name'])
            assert utils.none_equal(all_artists[artist_id]['date_created'], artist_info['date_created'])
    return InfoFrame.from_dict(all_artists, orient='index')


def get_albums_info(all_tracks_info: pandas.DataFrame) -> InfoFrame:
    all_albums = {}
    for track_id, track_info in progressbar(all_tracks_info.iterrows(), length=len(all_tracks_info)):
        album_info = track_info['album'].to_dict()
        album_id = album_info.pop('id')
        if album_id not in all_albums:
            all_albums[album_id] = album_info
        else:
            assert utils.none_equal(all_albums[album_id]['title'], album_info['title'])
            assert utils.none_equal(all_albums[album_id]['date_created'], album_info['date_created'])
    return InfoFrame.from_dict(all_albums, orient='index')


def main():
    (out_dir := FMA_DIR / 'metadata').mkdir(exist_ok=True)
    all_tracks_info = load_all_tracks_info()
    artists_info = get_artists_info(all_tracks_info)
    artists_info.to_tsv(out_dir / 'artists_info.tsv')
    artists_info.to_pkl(out_dir / 'artists_info.pkl')
    albums_info = get_albums_info(all_tracks_info)
    albums_info.to_tsv(out_dir / 'albums_info.tsv')
    albums_info.to_pkl(out_dir / 'albums_info.pkl')


if __name__ == '__main__':
    main()
