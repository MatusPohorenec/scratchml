import numpy as np
from typing import Tuple, Union, List
from abc import ABC

class BaseScaler(ABC):
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.array) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, y: np.array) -> np.ndarray:
        pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass

class MinMaxScaler(BaseScaler):
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        copy: bool = True,
        clip: bool = False
    ) -> None:
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.feature_names_in_ = None

    def fit(
        self,
        X: Union[np.ndarray, List],
        y: np.array = None
    ) -> None:
        if isinstance(X, list):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            pass
        else:
            return TypeError

        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.data_max_ = X.max(axis=0)
        self.data_min_ = X.min(axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.data_max_ - self.data_min_)
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.data_range_ = self.data_max_ - self.data_min_

    def transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        if self.clip:
            X_scaled = np.clip(
                a=X_scaled,
                a_min=self.feature_range[0],
                a_max=self.feature_range[1]
            )

        if self.copy:
            X = X_scaled.copy()
            return X
        else:
            return X_scaled
    
    def inverse_transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        if isinstance(X, list):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            pass
        else:
            return TypeError
        
        Xt = ((X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]))
        Xt *= (self.data_max_ - self.data_min_)
        Xt += self.data_min_
        return Xt

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.array = None
    ) -> np.ndarray:
        self.fit(X=X, y=y)
        return self.transform(X=X)
    
class StandardScaler(BaseScaler):
    def __init__(
        self,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True
    ) -> None:
        self.copy = copy
        self.with_std = with_std
        self.with_mean = with_mean
        self.scale_ = None
        self.mean_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def fit(
        self,
        X: Union[np.ndarray, List],
        y: np.array = None
    ) -> None:
        if isinstance(X, list):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            pass
        else:
            return TypeError

        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]

        if not self.with_mean and not self.with_std:
            self.var_ = None
            self.mean_ = None
        else:
            self.var_ = X.var(axis=0)
            self.mean_ = X.mean(axis=0)
        
        if not self.with_mean:
            self.mean_ = None
        else:
            self.mean_ = X.mean(axis=0)

        if not self.with_std:
            self.scale_ = None
            self.var_ = None
            
        else:
            self.std = np.std(X, axis=0)
            self.var_[self.var_ == 0] = 1
            self.scale_ = np.sqrt(self.var_)

    def transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        if isinstance(X, list):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            pass
        else:
            return TypeError
               
        if not self.with_std:
            self.std = 1

        if not self.with_mean:
            X_ = X / self.std
        else:
            X_ = (X - self.mean_) / self.std

        if self.copy:
            return X_
        else:
            X = X_.copy()
            return X
    
    def inverse_transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        if isinstance(X, list):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            pass
        else:
            return TypeError
        
        if not self.with_std:
            self.std = 1

        if not self.with_mean:
            Xt = (X * self.std)
        else:
            Xt = (X * self.std) + self.mean_

        return Xt

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.array = None
    ) -> np.ndarray:
        self.fit(X=X, y=y)
        return self.transform(X=X)