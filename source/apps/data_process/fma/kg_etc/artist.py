import collections
import pickle
import pandas

import dl
from apps import N

import utils
from apps.data_process import SRC_MUSIC_TAGGING_DIR, FMA_DIR
from apps.data_process.fma.kg_etc import FMA_KG_DIR
from dl.data import InfoFrame
from utils.log import progressbar

log = utils.log.get_logger()

filtered_track_ids = pickle.load(
    open(SRC_MUSIC_TAGGING_DIR / 'track_ids_in_key_order.p', 'rb'))

fma_all_data = dl.x_dataset.XDataset.load_from_disk(FMA_DIR / "fma_all_data")
filtered_track_data = fma_all_data.filter_by_ids(N.track_id, filtered_track_ids)

# genre_id_inst_id_conf = utils.file.load_pkl(data_dir / "genre_id_inst_id_conf.pkl")
genre_info = InfoFrame.load(FMA_DIR / 'label_map' / "157_genres.pkl")
genre_ids_empty_dict = {genre_id: 0 for genre_id in genre_info.index}
genre_id_artist_id_sum = collections.defaultdict(genre_ids_empty_dict.copy)

for artist_id, genre_ids in progressbar(
        zip(filtered_track_data[N.artist_id], filtered_track_data[N.genre_ids]), length=len(filtered_track_data)):
    for genre_id in genre_ids:
        genre_id_artist_id_sum[artist_id][genre_id] += 1

genre_id_artist_id_sum = pandas.DataFrame(genre_id_artist_id_sum)
utils.file.save_to_pkl(genre_id_artist_id_sum, FMA_KG_DIR / "genre_id_artist_id_sum.pkl")

genre_id_artist_id_conf = genre_id_artist_id_sum.div(genre_id_artist_id_sum.sum(axis=0), axis=1)
utils.file.save_to_pkl(genre_id_artist_id_conf, FMA_KG_DIR / "genre_id_artist_id_conf.pkl")

artists_info = InfoFrame.load(FMA_DIR / 'metadata' / "artists_info.pkl")

genre_name_artist_name_sum = genre_id_artist_id_sum.rename(
    index=lambda g_id: genre_info.id2name(g_id),
    columns=lambda a_id: artists_info.id2name(a_id)
)
utils.file.save_to_pkl(genre_name_artist_name_sum, FMA_KG_DIR / "genre_name_artist_name_sum.pkl")
genre_name_artist_name_sum.T.to_csv(FMA_KG_DIR / "genre_name_artist_name_sum.csv")

print("done")
