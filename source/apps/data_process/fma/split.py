import random
from collections import defaultdict

import utils.file
from apps import N
from apps.data_process import FMA_DIR, SRC_MUSIC_TAGGING_DIR
from apps.data_process.fma import load_all_tracks_info
from dl import x_dataset

log = utils.log.get_logger()


def get_classify_split(set_name):
    all_tracks_info = load_all_tracks_info()
    tracks_set = all_tracks_info[all_tracks_info['set', 'subset'] <= set_name]
    dataset = x_dataset.XDataset.load_from_disk(FMA_DIR / f"fma_all_data")
    all_ids = set(dataset[N.track_id])
    return {
        N.training: list(set(tracks_set[tracks_set['set', 'split'] == 'training'].index.to_list()) & all_ids),
        N.validation: list(set(tracks_set[tracks_set['set', 'split'] == 'validation'].index.to_list()) & all_ids),
        N.testing: list(set(tracks_set[tracks_set['set', 'split'] == 'test'].index.to_list()) & all_ids),
    }


def get_balanced_classify_split(set_name):
    all_tracks_info = load_all_tracks_info()
    original_split = get_classify_split(set_name)
    training_set = original_split[N.training]
    original_genres = all_tracks_info.loc[training_set, ('track', 'genre_top')]
    genres_multiple = original_genres.value_counts()
    genres_multiple = genres_multiple.max() / genres_multiple
    balanced_training_set = []
    for t_id, genre in original_genres.items():
        multiple = genres_multiple[genre]
        balanced_training_set.extend([t_id] * int(multiple))
        if random.random() < multiple % 1:
            balanced_training_set.append(t_id)

    original_split[N.training] = balanced_training_set
    return original_split


def get_ismir_19_split(split_name):
    tracks_set = load_all_tracks_info()
    available_gids = set(tracks_set.index)
    a_split, b_split, c_split, ab_split = map(
        lambda s_name: utils.file.load_pkl(SRC_MUSIC_TAGGING_DIR / f'track_keys_{s_name}_{split_name}.p'),
        ('A', 'B', 'C', 'AB'))

    tracks_id_in_key_order = utils.file.load_pkl(SRC_MUSIC_TAGGING_DIR / f'track_ids_in_key_order.p')
    a_train, a_eval, b_train, b_eval, c_eval = map(
        lambda s: [tracks_id_in_key_order[kid] for kid in s],
        (a_split[0], a_split[1], b_split[0], b_split[1], c_split))
    assert len(set(a_train) & available_gids) == len(a_train)
    return {
        N.training: a_train + a_eval,
        N.validation: b_train + b_eval,
        N.testing: c_eval,
    }


def get_random_zsl_split(seen_cls_num=124):
    tracks_set = load_all_tracks_info()
    from dl.data import InfoFrame
    genre_info = InfoFrame.load(FMA_DIR / 'label_map' / "157_genres.pkl")
    seen_genres = set(random.sample(genre_info.ids, seen_cls_num))
    unseen_genres = set(genre_info.ids) - seen_genres
    all_ids = utils.file.load_pkl(SRC_MUSIC_TAGGING_DIR / f'track_ids_in_key_order.p')
    split_conf = {N.training: [], N.validation: [], N.testing: []}
    for t_id in all_ids:
        genres = set(tracks_set.loc[t_id, ('track', 'genres')])
        if genres.issubset(seen_genres):
            split_conf[N.training].append(t_id)
        elif genres.issubset(unseen_genres):
            split_conf[N.testing].append(t_id)
        else:
            split_conf[N.validation].append(t_id)
    return split_conf, seen_genres


def get_seen_genres(split_conf):
    tracks_set = load_all_tracks_info()
    split_genres = defaultdict(set)
    for split_name, split_gids in split_conf.items():
        for genres in tracks_set.loc[split_gids, ('track', 'genres')]:
            split_genres[split_name].update(genres)

    assert len(split_genres[N.training] & split_genres[N.testing]) == 0
    return split_genres[N.training]


def main():
    (split_dir := FMA_DIR / "split").mkdir(exist_ok=True)

    split_name = 'medium'
    split_conf = get_classify_split(split_name)
    utils.file.save_to_pkl(split_conf, split_dir / f"classification-{split_name}.pkl")
    split_conf = get_balanced_classify_split(split_name)
    utils.file.save_to_pkl(split_conf, split_dir / f"balanced_classification-{split_name}.pkl")

    # split_name = 'TRSPP_TGSPP'  # 'TRSPP_TGSPP' , 'TRS02_TGS02'
    # split_conf = get_ismir_19_split(split_name)
    #
    # # split_name = 'TRSRd'
    # # split_conf, seen_gs = get_random_zsl_split(seen_cls_num=124)
    #
    # utils.file.save_to_pkl(split_conf, split_dir / f"zsl-ismir_19-{split_name}.pkl")
    #
    # split_conf = utils.file.load_pkl(split_dir / f"zsl-ismir_19-{split_name}.pkl")
    # seen_genres = get_seen_genres(split_conf)
    #
    # genre_info_data = utils.file.load_pkl(FMA_DIR / "label_map" / "157_genres.pkl")
    # genre_info_data[N.seen] = genre_info_data.index.isin(seen_genres)
    # utils.file.save_to_pkl(genre_info_data, FMA_DIR / "label_map" / f"157_genres.ismir_19-{split_name}.pkl")

    print("done")


if __name__ == '__main__':
    main()
