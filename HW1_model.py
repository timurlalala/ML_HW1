from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import re
from typing import Collection, Any, Union

class DataPreprocessor(TransformerMixin):
    """
    Класс, реализующий обучаемые функции для предобработки данных
    """

    def __init__(self, features_from_num_to_cat: Collection[str] = None) -> None:
        self.features_from_num_to_cat = features_from_num_to_cat

    def fit(self, X: pd.DataFrame, y: Any = None) -> None:
        X_cat = X.select_dtypes(exclude='number')
        X_num = X.select_dtypes(include='number')
        if self.features_from_num_to_cat:
            X_cat = pd.concat([X_cat, X_num[self.features_from_num_to_cat]], axis=1)
            X_num = X_num.drop(columns=self.features_from_num_to_cat)

        self.OHE = OneHotEncoder(drop='first', handle_unknown="ignore", sparse_output=False)
        self.OHE.fit(X_cat)

        self.SScaler = StandardScaler()
        self.SScaler.fit(X_num)

        self.feature_names_in_ = np.concatenate([self.OHE.feature_names_in_, self.SScaler.feature_names_in_], axis=0)
        return None

    def transform(self, X: pd.DataFrame, y: Any = None) -> np.ndarray:
        X = X[self.feature_names_in_]
        X_cat = X.select_dtypes(exclude='number')
        X_num = X.select_dtypes(include='number')
        if self.features_from_num_to_cat:
            X_cat = pd.concat([X_cat, X_num[self.features_from_num_to_cat]], axis=1)
            X_num = X_num.drop(columns=self.features_from_num_to_cat)

        cat = self.OHE.transform(X_cat)
        num = self.SScaler.transform(X_num)

        return np.concatenate([cat, num], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params) -> np.ndarray:
        self.fit(X, y)
        Xt = self.transform(X, y)
        return Xt



num = re.compile(r'\d+.??\d*')
def get_number(s: Union[str, pd.NA]) -> str:
    if not pd.isna(s):
        m = re.match(num, s)
        if m:
            return m.group().replace(',', '.')
    return '0'

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит столбцы к предопределенному формату"""
    df = df.copy()
    for column in ('mileage', 'engine', 'max_power'):
        df[column] = pd.to_numeric(df[column].apply(lambda x: get_number(x)))
    df['name'] = df['name'].apply(lambda x: x.upper().split()[0])
    df = df.convert_dtypes()
    return df
