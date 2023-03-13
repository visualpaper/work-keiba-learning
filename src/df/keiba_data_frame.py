import pandas as pd

from typing import List
from dataclasses import dataclass
from pandas.core.frame import DataFrame


@dataclass
class KeibaDataFrame:
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

    def filter_datas(self, target_classes: List[int]):
        self._df.query("race_class_id in @target_classes", inplace=True)

    def setting_datas(self):
        # date 項目をエポック秒に変換
        self._df["date_num"] = self._df.apply(self._to_date_num, axis=1)
        self._df["date_num"] = self._df["date_num"].astype(int)
        self._df.drop(["date"], axis=1, inplace=True)

    def _to_date_num(self, row) -> int:
        d = pd.to_datetime(row["date"], format="%Y-%m-%d")
        localized = d.tz_localize('Asia/Tokyo')
        return int(localized.timestamp())

    def setting_types(self):
        # int 型を明示化
        self._df[
            [
                "track_id",
                "date_num",
                "round",
                "distance",
                "race_weather_id",
                "shipping_time_hour",
                "shipping_time_minute",
                "shipping_time_seconds",
                "race_class_id",
                "number",
                "frame",
                "horse_id_id",
                "barei",
                "mother_horse_id_id",
                "mother_grand_mother_horse_id_id",
                "mother_grand_mother_grand_mother_horse_id_id",
                "mother_grand_mother_grand_father_horse_id_id",
                "mother_grand_father_horse_id_id",
                "mother_grand_father_grand_mother_horse_id_id",
                "mother_grand_father_grand_father_horse_id_id",
                "father_horse_id_id",
                "father_grand_mother_horse_id_id",
                "father_grand_mother_grand_mother_horse_id_id",
                "father_grand_mother_grand_father_horse_id_id",
                "father_grand_father_horse_id_id",
                "father_grand_father_grand_mother_horse_id_id",
                "father_grand_father_grand_father_horse_id_id",
                "gender_type",
                "jockey_id",
                "popular",
                "weight",
                "weight_change",
                "trainer_id",
                "owner_id",
                "rank",
            ]
        ] = self._df[
            [
                "track_id",
                "date_num",
                "round",
                "distance",
                "race_weather_id",
                "shipping_time_hour",
                "shipping_time_minute",
                "shipping_time_seconds",
                "race_class_id",
                "number",
                "frame",
                "horse_id_id",
                "barei",
                "mother_horse_id_id",
                "mother_grand_mother_horse_id_id",
                "mother_grand_mother_grand_mother_horse_id_id",
                "mother_grand_mother_grand_father_horse_id_id",
                "mother_grand_father_horse_id_id",
                "mother_grand_father_grand_mother_horse_id_id",
                "mother_grand_father_grand_father_horse_id_id",
                "father_horse_id_id",
                "father_grand_mother_horse_id_id",
                "father_grand_mother_grand_mother_horse_id_id",
                "father_grand_mother_grand_father_horse_id_id",
                "father_grand_father_horse_id_id",
                "father_grand_father_grand_mother_horse_id_id",
                "father_grand_father_grand_father_horse_id_id",
                "gender_type",
                "jockey_id",
                "popular",
                "weight",
                "weight_change",
                "trainer_id",
                "owner_id",
                "rank",
            ]
        ].astype(
            int
        )

        # bool 型を明示化
        self._df[
            [
                "type_turf",
                "type_dirt",
                "type_hundle",
                "clockwise_inner",
                "clockwise_outer",
                "clockwise_left",
                "clockwise_right",
                "entry_condition_shitei",
                "entry_condition_kongou",
                "entry_condition_tokushi",
                "entry_condition_kokusai",
                "entry_condition_barei",
                "entry_condition_bettei",
                "entry_condition_teiryo",
                "entry_condition_hande",
            ]
        ] = self._df[
            [
                "type_turf",
                "type_dirt",
                "type_hundle",
                "clockwise_inner",
                "clockwise_outer",
                "clockwise_left",
                "clockwise_right",
                "entry_condition_shitei",
                "entry_condition_kongou",
                "entry_condition_tokushi",
                "entry_condition_kokusai",
                "entry_condition_barei",
                "entry_condition_bettei",
                "entry_condition_teiryo",
                "entry_condition_hande",
            ]
        ].astype(
            bool
        )

        # float 型を明示化
        self._df[["impost", "odds", "prize"]] = self._df[
            ["impost", "odds", "prize"]
        ].astype(float)

        # Nullable float 型を明示化
        self._df[["time", "furlong"]] = self._df[["time", "furlong"]].astype(
            pd.Float64Dtype()
        )

        self._df.drop(
            ["popular", "time", "furlong", "rank", "prize"],
            axis=1,
            inplace=True,
        )

    def groups(self):
        sorted_df = self._df.sort_values(
            ["date_num", "track_id", "round"], ascending=True
        )

        for _, v in sorted_df.groupby(["date_num", "track_id", "round"], sort=False):
            yield v
