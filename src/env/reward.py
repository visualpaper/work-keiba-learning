import pandas as pd

from dataclasses import dataclass
from pandas.core.frame import DataFrame

from df.result_frame import ResultFrame
from env.actions import Action


@dataclass
class Reward:
    _rdf: DataFrame

    @classmethod
    def initialize(cls, result_path: str):

        # Result Data 読み込み
        rdf = ResultFrame.from_data(result_path)
        rdf.setting_datas()
        rdf.setting_types()
        return cls(rdf.df)

    def _get_result_target(self, key: DataFrame) -> DataFrame:
        return self._rdf.loc[
            (self._rdf["track_id"] == int(key["track_id"].values[0]))
            & (self._rdf["date_num"] == int(key["date_num"].values[0]))
            & (self._rdf["round"] == int(key["round"].values[0]))
        ]

    def _calc_result_action_reward(
        self, result_target: DataFrame, select_horse: pd.Series
    ) -> int:
        result = result_target[(result_target["number"] == int(select_horse["number"]))]

        if len(result) == 0:
            return -1000

        return int((float(result["odds"].values[0]) * 1000) - 1000)

    def calc(self, race: DataFrame, action: int) -> int:
        df = race.sort_values(["pred"], ascending=[False])
        result_target = self._get_result_target(df.head(1))

        '''
        action
        0: 1 着予想の馬を買う
        1: 2 着予想の馬を買う
        2: 3 着予想の馬を買う
        3: 4 着予想の馬を買う
        4: 5 着予想の馬を買う
        6: 買わない
        '''
        action0_reward = self._calc_result_action_reward(result_target, df.iloc[0])
        action1_reward = self._calc_result_action_reward(result_target, df.iloc[1])
        action2_reward = self._calc_result_action_reward(result_target, df.iloc[2])
        action3_reward = self._calc_result_action_reward(result_target, df.iloc[3])
        try:
            action4_reward = self._calc_result_action_reward(result_target, df.iloc[4])
        except Exception:
            action4_reward = -1000

        if action == Action.RANK_ONE_HORSE.value:
            reward = action0_reward

        elif action == Action.RANK_TWO_HORSE.value:
            reward = action1_reward

        elif action == Action.RANK_THREE_HORSE.value:
            reward = action2_reward

        elif action == Action.RANK_FOUR_HORSE.value:
            reward = action3_reward

        elif action == Action.RANK_FIVE_HORSE.value:
            reward = action4_reward

        else:
            if (
                action0_reward >= 0
                or action1_reward >= 0
                or action2_reward >= 0
                or action3_reward >= 0
                or action4_reward >= 0
            ):
                reward = -2500
            else:
                reward = 2500

        return reward
