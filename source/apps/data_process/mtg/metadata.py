import pandas

import utils.file
from apps.data_process import SRC_MTG_DIR, MTG_DIR
from dl.data import InfoFrame

log = utils.log.get_logger()


def read_meta_data_raw():
    num_com = 8
    lines_buffer = []

    with (SRC_MTG_DIR / "raw.meta.tsv").open('r') as reader:
        for i, line in enumerate(reader.readlines()):
            ls = line.rstrip().split('\t')
            assert num_com == len(ls)
            lines_buffer.append(ls)

    return pandas.DataFrame(lines_buffer[1:], columns=lines_buffer[0])


def get_artists_info(raw_data: pandas.DataFrame) -> InfoFrame:
    artists_info = {}
    ids = raw_data['ARTIST_ID'].unique()
    for id_str in ids:
        a_id = int(id_str.split('_')[-1])
        raw = raw_data[raw_data['ARTIST_ID'] == id_str]
        names = raw['ARTIST_NAME'].unique()
        names = set(utils.remove_special_char(name) for name in names)
        if len(names) == 1:
            artists_info[a_id] = names.pop()
        else:
            name = utils.remove_special_char(raw['ARTIST_NAME'].mode()[0])
            name = f"{name} ({'|'.join(n for n in names if n != name)})"
            artists_info[a_id] = name
            log.warning(f"artist {a_id} has multiple names: \t{name}")
    return InfoFrame.from_dict(artists_info, orient='index', columns=['name'])


def get_albums_info(raw_data: pandas.DataFrame) -> InfoFrame:
    albums_info = {}
    ids = raw_data['ALBUM_ID'].unique()
    for id_str in ids:
        a_id = int(id_str.split('_')[-1])
        raw = raw_data[raw_data['ALBUM_ID'] == id_str]

        release_dates = raw['RELEASEDATE'].unique()
        if len(release_dates) > 1:
            release_date = raw['RELEASEDATE'].mode()[0]
            log.warning(f"album {a_id} use {release_date} replace {release_dates}")
        else:
            release_date = release_dates[0]

        # release_date = time.strptime(release_date, "%Y-%m-%d")

        names = set(utils.remove_special_char(name) for name in raw['ALBUM_NAME'].unique())
        if len(names) == 1:
            albums_info[a_id] = (names.pop(), release_date)
        else:
            name = utils.remove_special_char(raw['ALBUM_NAME'].mode()[0])
            name = f"{name} ({'|'.join(n for n in names if n != name)})"
            albums_info[a_id] = (name, release_date)
            log.warning(f"artist {a_id} has multiple names: \t{name}")

    return InfoFrame.from_dict(albums_info, orient='index', columns=['name', 'release_date'])


def main():
    (out_dir := MTG_DIR / 'metadata').mkdir(exist_ok=True)
    meta_data_raw = read_meta_data_raw()
    artists_info = get_artists_info(meta_data_raw)
    artists_info.to_tsv(out_dir / 'artists_info.tsv')
    artists_info.to_pkl(out_dir / 'artists_info.pkl')
    albums_info = get_albums_info(meta_data_raw)
    albums_info.to_tsv(out_dir / 'albums_info.tsv')
    albums_info.to_pkl(out_dir / 'albums_info.pkl')


if __name__ == '__main__':
    main()
