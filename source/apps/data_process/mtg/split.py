import utils.file
from apps import N
from apps.data_process import SRC_MTG_DIR, MTG_DIR

log = utils.log.get_logger()


def get_split(file_name):
    track_ids = []
    with file_name.open('r') as reader:
        assert reader.readline().split('\t')[0] == 'TRACK_ID'
        for line in reader.readlines():
            track_ids.append(int(line.split('\t')[0][6:]))
    return track_ids


def main():
    task_prefix = 'autotagging_genre'
    split_name = 'split-0'

    split_conf = {
        N.training: get_split(SRC_MTG_DIR / "splits" / split_name / f'{task_prefix}-train.tsv'),
        N.validation: get_split(SRC_MTG_DIR / "splits" / split_name / f'{task_prefix}-validation.tsv'),
        N.testing: get_split(SRC_MTG_DIR / "splits" / split_name / f'{task_prefix}-test.tsv'),
    }

    (split_dir := MTG_DIR / "split").mkdir(exist_ok=True)

    utils.file.save_to_pkl(split_conf, split_dir / f"{task_prefix}-{split_name}.pkl")


if __name__ == '__main__':
    main()
