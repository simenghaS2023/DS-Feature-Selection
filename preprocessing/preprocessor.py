import json
import re
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

DATA_ROOT = "data"


class Preprocessor:
    def __init__(self, config_path="app_config.json"):
        self.config = json.load(open(config_path))
        self.data_dir = os.path.join(DATA_ROOT, self.config["data_dir"])
        self.data_indices = self.config["data_indices"]
        self.data_index_pattern = re.compile(r'T(\d+).csv')
        
    def preprocess(self) -> Tuple[pd.DataFrame,np.ndarray, pd.DataFrame, np.ndarray]:
        df = self._load_data()
        train_X, train_y, test_X, test_y = self._train_test_split(df)
        
        return train_X, train_y, test_X, test_y

    def _load_data(self):
        data_files = [
            os.path.join(self.data_dir, filename)
            for filename in os.listdir(self.data_dir)
            if self.data_index_pattern.search(filename) and int(self.data_index_pattern.search(filename).group(1)) in self.data_indices
        ]
        
        dfs = [pd.read_csv(file, comment='#').rename(str.strip, axis="columns") for file in data_files]
        df = pd.concat(dfs)
        df = df.sample(frac=1).reset_index(drop=True)
        return df
        
    def _train_test_split(self, df:pd.DataFrame, test_size=0.2):
        train, test = train_test_split(df, test_size=test_size, stratify=df.iloc[:, -1])
        
        train_X = train.iloc[:, :-1]
        train_y = train.iloc[:, -1]

        test_X = test.iloc[:, :-1]
        test_y = test.iloc[:, -1]

        label_encoder = LabelEncoder()
        label_encoder.fit(train_y)

        train_y = label_encoder.transform(train_y)
        test_y = label_encoder.transform(test_y)
        
        return train_X, train_y, test_X, test_y
        
        
        
        
