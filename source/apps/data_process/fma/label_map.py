import pandas

import utils
from apps import N
from apps.data_process import FMA_DIR, SRC_MUSIC_TAGGING_DIR
from apps.data_process.fma import SRC_FMA_META_DIR
from dl.data import InfoFrame

log = utils.log.get_logger()


def read_all_genres():
    genres_df = pandas.read_csv(SRC_FMA_META_DIR / "genres.csv", index_col=0)
    genres_df.drop(columns=['#tracks', ], inplace=True)
    genres_df.rename(columns={'title': N.name, }, inplace=True)
    return genres_df


def add_zsl_tagging_filter(genres_df):
    filtered_ids = utils.file.load_pkl(SRC_MUSIC_TAGGING_DIR / "tag_ids_in_key_order.p")
    genres_df[N.ismir19_genre] = genres_df.index.isin(filtered_ids)
    return genres_df


def add_top_genre_filter(genres_df):
    genres_df[N.top_genre] = (genres_df['parent'] == 0)
    return genres_df


def main():
    (out_dir := FMA_DIR / "label_map").mkdir(exist_ok=True)
    genres_df = read_all_genres()
    genres_df = add_zsl_tagging_filter(genres_df)
    genres_df = add_top_genre_filter(genres_df)
    genres_map = InfoFrame(genres_df)

    genres_map.to_pkl(out_dir / "all_genres.pkl")

    genres_map.to_tsv(out_dir / "all_genres.tsv")
    genres157_map = genres_map.filter_by_column(N.ismir19_genre)
    genres157_map.to_tsv(out_dir / "157_genres.tsv")
    genres157_map.to_pkl(out_dir / "157_genres.pkl")
    top_genres_map = genres_map.filter_by_column(N.top_genre)
    top_genres_map.to_tsv(out_dir / "top_genres.tsv")
    print("done")


if __name__ == '__main__':
    main()
