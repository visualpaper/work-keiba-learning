from typing import List
import pandas as pd

from dataclasses import dataclass
from pandas.core.frame import DataFrame


@dataclass
class ResultFrame:
    _df: DataFrame

    @classmethod
    def from_data(cls, path: str):
        return cls(pd.read_csv(path, encoding="utf-8"))

    @classmethod
    def from_datas(cls, paths: List[str]):
        datas = [pd.read_csv(path, encoding="utf-8") for path in paths]
        return cls(pd.concat(datas))

    @property
    def df(self) -> DataFrame:
        return self._df

    def setting_datas(self):
        # date 項目を分解
        self._df["date_num"] = self._df.apply(self._to_date_num, axis=1)
        self._df["date_num"] = self._df["date_num"].astype(int)
        self._df.drop(["date"], axis=1, inplace=True)

    def _to_date_num(self, row) -> int:
        d = pd.to_datetime(row["date"], format="%Y-%m-%d")
        localized = d.tz_localize('Asia/Tokyo')
        return int(localized.timestamp())

    def setting_types(self):
        # int 型を明示化
        self._df[["track_id", "date_num", "round", "number", "number2"]] = self._df[
            ["track_id", "date_num", "round", "number", "number2"]
        ].astype(int)

        # float 型を明示化
        self._df[["odds"]] = self._df[["odds"]].astype(float)
