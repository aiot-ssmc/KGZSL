import pandas

import dl
import utils
from apps import N
from apps.data_process import MTG_DIR
from apps.data_process.mtg.kg_etc import genre_info, artists_info, instrument_info, MTG_KG_DIR
from utils.log import progressbar

all_data = dl.x_dataset.XDataset.load_from_disk(MTG_DIR / "mtg_all_data")


def get_conf_sums():
    genre_id_artist_id_sum = pandas.DataFrame(0, index=genre_info.index, columns=artists_info.index)
    genre_id_instrument_id_sum = pandas.DataFrame(0, index=genre_info.index, columns=instrument_info.index)
    for tid, artist_id, genre_ids, instrument_ids in progressbar(
            zip(all_data[N.track_id], all_data[N.artist_id], all_data[N.genre_ids], all_data[N.instrument_ids]),
            length=len(all_data)):
        for genre_id in genre_ids:
            genre_id_artist_id_sum[artist_id][genre_id] += 1
            for instrument_id in instrument_ids:
                genre_id_instrument_id_sum[instrument_id][genre_id] += 1

    return genre_id_artist_id_sum, genre_id_instrument_id_sum


def main():
    genre_id_artist_id_sum, genre_id_instrument_id_sum = get_conf_sums()
    utils.file.save_to_pkl(genre_id_artist_id_sum, MTG_KG_DIR / "genre_id_artist_id_sum.pkl")
    utils.file.save_to_pkl(genre_id_instrument_id_sum, MTG_KG_DIR / "genre_id_instrument_id_sum.pkl")
    genre_id_artist_id_conf = genre_id_artist_id_sum.div(genre_id_artist_id_sum.sum(axis=0), axis=1).fillna(0)
    utils.file.save_to_pkl(genre_id_artist_id_conf, MTG_KG_DIR / "genre_id_artist_id_conf.pkl")
    genre_id_instrument_id_conf = genre_id_instrument_id_sum.div(genre_id_instrument_id_sum.sum(axis=0), axis=1)
    utils.file.save_to_pkl(genre_id_instrument_id_conf, MTG_KG_DIR / "genre_id_instrument_id_conf.pkl")

    genre_name_artist_name_conf = genre_id_artist_id_conf.rename(
        index=lambda g_id: genre_info.id2name(g_id),
        columns=lambda a_id: artists_info.id2name(a_id)
    )
    genre_name_artist_name_conf.T.to_csv(MTG_KG_DIR / "genre_name_artist_name_conf.csv")

    genre_name_instrument_name_conf = genre_id_instrument_id_conf.rename(
        index=lambda g_id: genre_info.id2name(g_id),
        columns=lambda i_id: instrument_info.id2name(i_id)
    )
    genre_name_instrument_name_conf.T.to_csv(MTG_KG_DIR / "genre_name_instrument_name_conf.csv")

    print("done")


if __name__ == '__main__':
    main()
