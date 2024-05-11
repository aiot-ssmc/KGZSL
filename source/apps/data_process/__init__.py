from apps.config import args

SRC_DIR = args.input / "source"

SRC_FMA_DIR = SRC_DIR / "Free_Music_Archive"

SRC_OPEN_MIC_DIR = SRC_DIR / "openmic"
# extract from openmic-2018-v1.0.0.tgz/openmic-2018

SRC_MUSIC_TAGGING_DIR = SRC_DIR / "zsl_music_tagging"
# extract from data_common.zip/data_common/fma/*
# data_common.zip from https://github.com/kunimi00/ZSL_music_tagging.git

SRC_MTG_DIR = SRC_DIR / "mtg-jamendo-dataset"

FMA_DIR = args.input / "fma"
MTG_DIR = args.input / "mtg"
