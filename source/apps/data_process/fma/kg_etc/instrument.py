import json
import pickle

import numpy
import numpy as np
import pandas as pd

import utils
from apps.data_process import SRC_OPEN_MIC_DIR, SRC_MUSIC_TAGGING_DIR

from apps.data_process.fma import load_all_tracks_info, SRC_FMA_META_DIR
from apps.data_process.fma.kg_etc import FMA_KG_DIR

log = utils.log.get_logger()

inst_id_to_inst_name_dict = json.load(open(SRC_OPEN_MIC_DIR / "class-map.json"))

data = numpy.load(SRC_OPEN_MIC_DIR / 'openmic-2018.npz', allow_pickle=True)
Y_true, Y_mask, sample_key = data['Y_true'], data['Y_mask'], data['sample_key']
song_key_by_inst_key_conf_matrix = Y_true.copy()
song_key_by_inst_key_conf_matrix[~Y_mask] = float('nan')

track_id_to_inst_conf_dict = dict()
for idx in range(len(sample_key)):
    t_id = int(sample_key[idx].split('_')[0])
    track_id_to_inst_conf_dict[t_id] = song_key_by_inst_key_conf_matrix[idx]

tracks = load_all_tracks_info()
features = pd.read_csv((SRC_FMA_META_DIR / 'features.csv'), index_col=0, header=[0, 1, 2])
echonest = pd.read_csv((SRC_FMA_META_DIR / 'echonest.csv'), index_col=0, header=[0, 1, 2])
genres = pd.read_csv((SRC_FMA_META_DIR / 'genres.csv'), index_col=0)

###########################################################################################################
# ref from https://github.com/kunimi00/ZSL_music_tagging/blob/master/scripts/data_prep_instrument_vector.py

numpy.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

print(tracks.shape, features.shape, echonest.shape)

tracks_large = tracks['set', 'subset'] <= 'large'
track_genres_top = tracks.loc[tracks_large, ('track', 'genre_top')].values.tolist()
track_genres = tracks.loc[tracks_large, ('track', 'genres')].values.tolist()
track_ids = tracks.loc[tracks_large].index
genre_titles = genres['title'].tolist()
genre_ids = genres.index.tolist()

track_ids_with_genres = []
track_id_to_genre_id_dict = {}
for i in range(len(track_genres)):
    if len(track_genres[i]) > 0:
        track_ids_with_genres.append(track_ids[i])
        track_id_to_genre_id_dict[track_ids[i]] = track_genres[i]
    else:
        continue

print('track_ids_with_genres', len(track_ids_with_genres))  # 104343

genre_ids.sort()

track_ids_with_inst = []
for _key in sample_key:
    track_ids_with_inst.append(int(_key.split('_')[0]))

print('track_ids_with_inst', len(track_ids_with_inst))  # 20000

# Here we used prefiltered tracks (19466) and genres (157)
prefiltered_track_ids_in_key_order = pickle.load(
    open(SRC_MUSIC_TAGGING_DIR / 'track_ids_in_key_order.p', 'rb'))
prefiltered_tag_ids_in_key_order = pickle.load(open(SRC_MUSIC_TAGGING_DIR / 'tag_ids_in_key_order.p', 'rb'))

track_key_to_genre_key_binary_matrix = []

for key, t_id in enumerate(prefiltered_track_ids_in_key_order):
    curr_binary = np.zeros(len(prefiltered_tag_ids_in_key_order))
    for curr_genre_id in track_id_to_genre_id_dict[t_id]:
        curr_binary[prefiltered_tag_ids_in_key_order.index(curr_genre_id)] = 1

    track_key_to_genre_key_binary_matrix.append(curr_binary)

track_key_to_genre_key_binary_matrix = np.array(track_key_to_genre_key_binary_matrix)

print('track_key_to_genre_key_binary_matrix shape ', track_key_to_genre_key_binary_matrix.shape)
# (19466, 157)
genre_key_to_track_key_binary_matrix = track_key_to_genre_key_binary_matrix.T
# (157, 19466)


genre_id_to_inst_conf_dict = {}

for genre_key in range(len(prefiltered_tag_ids_in_key_order)):
    genre_id = prefiltered_tag_ids_in_key_order[genre_key]

    inst_conf_list = []

    curr_track_keys = np.argwhere(genre_key_to_track_key_binary_matrix[genre_key] == 1).squeeze()

    for _track_key in curr_track_keys:
        _track_id = prefiltered_track_ids_in_key_order[_track_key]
        _curr_track_inst_conf = track_id_to_inst_conf_dict[_track_id]

        inst_conf_list.append(_curr_track_inst_conf)
    inst_conf = numpy.stack(inst_conf_list)
    genre_id_to_inst_conf_dict[genre_id] = numpy.nanmean(inst_conf, axis=0)

df_genre_id_inst_id_conf = pd.DataFrame(genre_id_to_inst_conf_dict).T
utils.file.save_to_pkl(df_genre_id_inst_id_conf, FMA_KG_DIR / "genre_id_inst_id_conf.pkl")
df_genre_name_inst_name_conf = df_genre_id_inst_id_conf.rename(
    index={i: title for i, title in zip(genre_ids, genre_titles)},
    columns={i: n for n, i in inst_id_to_inst_name_dict.items()})
utils.file.save_to_pkl(df_genre_name_inst_name_conf, FMA_KG_DIR / "genre_name_inst_name_conf.pkl")
df_genre_name_inst_name_conf.T.to_csv(FMA_KG_DIR / "genre_name_inst_name_conf.csv")
print("done")
