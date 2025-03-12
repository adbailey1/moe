import numpy as np
import librosa


class FeatureExtractor:
    def __init__(self, rate):
        self.rate = rate

    def get_features(self, features_to_use, data):
        data_features = None
        accepted_features_to_use = 'mfcc'
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError(f"{features_to_use} not in"
                                      f" {accepted_features_to_use}!")
        if features_to_use in 'mfcc':
            data_features = self.get_mfcc(data, 26)

        return data_features

    def get_mfcc(self, data, n_mfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(y=x,
                                             sr=self.rate,
                                             n_mfcc=n_mfcc)
            return mfcc_data

        data_features = np.apply_along_axis(_get_mfcc, 1, data)
        return data_features
