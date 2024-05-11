from apps.data_process import MTG_DIR
from dl.data import InfoFrame

(MTG_KG_DIR := MTG_DIR / "knowledge_graph").mkdir(exist_ok=True)

artists_info = InfoFrame.load(MTG_DIR / 'metadata' / "artists_info.pkl")
instrument_info = InfoFrame.load(MTG_DIR / 'label_map' / "instrument.pkl")
genre_info = InfoFrame.load(MTG_DIR / 'label_map' / "genre.pkl")
