import time

import tqdm
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier

from apps.data_process.fma import load_all_tracks_info, SRC_FMA_META_DIR

tracks = load_all_tracks_info()

features = pd.read_csv(SRC_FMA_META_DIR / 'features.csv', index_col=0, header=[0, 1, 2])
features = pd.concat({
    'mfcc+rmse': pd.concat([features['mfcc'], features['rmse']], axis=1),
    'mfcc': features['mfcc'],
    'rmse': features['rmse']}, axis=1)

echonest = pd.read_csv(SRC_FMA_META_DIR / 'echonest.csv', index_col=0, header=[0, 1, 2])

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks.shape, features.shape, echonest.shape

subset = tracks.index[tracks['set', 'subset'] <= 'medium']

assert subset.isin(tracks.index).all()
assert subset.isin(features.index).all()

features_all = features.join(echonest, how='inner').sort_index(axis=1)
print('Not enough Echonest features: {}'.format(features_all.shape))

tracks = tracks.loc[subset]
features_all = features.loc[subset]

tracks.shape, features_all.shape

train = tracks.index[tracks['set', 'split'] == 'training']
val = tracks.index[tracks['set', 'split'] == 'validation']
test = tracks.index[tracks['set', 'split'] == 'test']

print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
# genres = list(tracks['track', 'genre_top'].unique())
print('Top genres ({}): {}'.format(len(genres), genres))
genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
print('All genres ({}): {}'.format(len(genres), genres))


def pre_process(tracks, features, columns, multi_label=False, verbose=False):
    if not multi_label:
        # Assign an integer value to each genre.
        enc = LabelEncoder()
        labels = tracks['track', 'genre_top']
        # y = enc.fit_transform(tracks['track', 'genre_top'])
    else:
        # Create an indicator matrix.
        enc = MultiLabelBinarizer()
        labels = tracks['track', 'genres_all']
        # labels = tracks['track', 'genres']

    # Split in training, validation and testing sets.
    y_train = enc.fit_transform(labels[train])
    y_val = enc.transform(labels[val])
    y_test = enc.transform(labels[test])
    X_train = features.loc[train, columns].values
    X_val = features.loc[val, columns].values
    X_test = features.loc[test, columns].values

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_val)
    scaler.transform(X_test)

    return y_train, y_val, y_test, X_train, X_val, X_test


def classifiers_features_tester(classifiers, feature_sets, multi_label=False):
    columns = list(classifiers.keys()).insert(0, 'dim')
    scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
    times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
    for fset_name, fset in tqdm.tqdm(feature_sets.items(), desc='features'):
        y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
        scores.loc[fset_name, 'dim'] = X_train.shape[1]
        for clf_name, clf in classifiers.items():  # tqdm_notebook(classifiers.items(), desc='classifiers', leave=False):
            t = time.process_time()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.loc[fset_name, clf_name] = score
            times.loc[fset_name, clf_name] = time.process_time() - t
    return scores, times


classifiers = {
    'LR': LogisticRegression(),
    'kNN': KNeighborsClassifier(n_neighbors=200),
    'SVCrbf': SVC(kernel='rbf'),
    # 'SVCpoly1': SVC(kernel='poly', degree=1),
    # 'linSVC1': SVC(kernel="linear"),
    # 'linSVC2': LinearSVC(),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    # 'DT': DecisionTreeClassifier(max_depth=5),
    # 'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'AdaBoost': AdaBoostClassifier(n_estimators=10),
    # 'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
    # 'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
    # 'NB': GaussianNB(),
    # 'QDA': QuadraticDiscriminantAnalysis(),
}

feature_sets = {
    #    'echonest_audio': ('echonest', 'audio_features'),
    #    'echonest_social': ('echonest', 'social_features'),
    #    'echonest_temporal': ('echonest', 'temporal_features'),
    #    'echonest_audio/social': ('echonest', ('audio_features', 'social_features')),
    #    'echonest_all': ('echonest', ('audio_features', 'social_features', 'temporal_features')),
}
for name in features.columns.levels[0]:
    feature_sets[name] = name
# feature_sets.update({
#     'mfcc/contrast': ['mfcc', 'spectral_contrast'],
#     'mfcc/contrast/chroma': ['mfcc', 'spectral_contrast', 'chroma_cens'],
#     'mfcc/contrast/centroid': ['mfcc', 'spectral_contrast', 'spectral_centroid'],
#     'mfcc/contrast/chroma/centroid': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid'],
#     'mfcc/contrast/chroma/centroid/tonnetz': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid',
#                                               'tonnetz'],
#     'mfcc/contrast/chroma/centroid/zcr': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr'],
#     'all_non-echonest': list(features.columns.levels[0])
# })


scores, times = classifiers_features_tester(classifiers, feature_sets)

print(scores)

print("done")
