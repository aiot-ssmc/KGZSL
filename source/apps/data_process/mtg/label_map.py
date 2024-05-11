from collections import defaultdict

import utils.file
from apps.data_process import SRC_MTG_DIR, MTG_DIR
from dl.data import InfoFrame

log = utils.log.get_logger()


def read_all_tags():
    tag_start = 5
    all_tags = set()

    with (SRC_MTG_DIR / "autotagging.tsv").open('r') as reader:
        assert 'TAGS' == reader.readline().rstrip().split('\t')[tag_start]
        for i, line in enumerate(reader.readlines()):
            ls = line.rstrip().split('\t')
            all_tags.update(ls[tag_start:])

    tag_dict = defaultdict(list)
    for tag in all_tags:
        cls, label = tag.split('---')
        tag_dict[cls].append(label)
    return tag_dict


def main():
    tags = read_all_tags()
    # tag_map = utils.file.load_json_as_dict(SRC_DIR / "tag_map.json")
    (out_dir := MTG_DIR / "label_map").mkdir(exist_ok=True)
    for tag_name, tag_labels in tags.items():
        label_map = InfoFrame.from_dict(
            {l_id: label for l_id, label in enumerate(sorted(tag_labels))},
            orient='index', columns=['name'])

        file_name = tag_name.replace('/', '_')
        label_map.to_tsv(out_dir / f"{file_name}.tsv")
        label_map.to_pkl(out_dir / f"{file_name}.pkl")
    print("done")


if __name__ == '__main__':
    main()
