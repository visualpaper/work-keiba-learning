from typing import List
import pandas as pd

from dataclasses import dataclass
from pandas.core.frame import DataFrame

from df.result_frame import ResultFrame
from env.actions import Action


@dataclass
class Reward:
    _rdf: DataFrame

    @classmethod
    def initialize(cls, result_path: List[str]):

        # Result Data 読み込み
        rdf = ResultFrame.from_datas(result_path)
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
        self,
        result_target: DataFrame,
        select_horse1: pd.Series,
        select_horse2: pd.Series,
    ) -> int:
        result1 = result_target[
            (result_target["number"] == int(select_horse1["number"]))
            & (result_target["number2"] == int(select_horse2["number"]))
        ]
        result2 = result_target[
            (result_target["number"] == int(select_horse2["number"]))
            & (result_target["number2"] == int(select_horse1["number"]))
        ]

        if len(result1) != 0:
            return int((float(result1["odds"].values[0]) * 1000))

        if len(result2) != 0:
            return int((float(result2["odds"].values[0]) * 1000))

        return 0

    def calc(self, race: DataFrame, action: Action) -> int:
        df = race.sort_values(["pred"], ascending=[False])
        result_target = self._get_result_target(df.head(1))

        '''
        action
        '''
        action0_reward = self._calc_result_action_reward(
            result_target, df.iloc[0], df.iloc[1]
        )
        action1_reward = self._calc_result_action_reward(
            result_target, df.iloc[0], df.iloc[2]
        )
        action2_reward = self._calc_result_action_reward(
            result_target, df.iloc[1], df.iloc[2]
        )
        # try:
        #     action3_reward = self._calc_result_action_reward(
        #         result_target, df.iloc[0], df.iloc[4]
        #     )
        # except Exception:
        #     action3_reward = 0

        # action4_reward = self._calc_result_action_reward(
        #     result_target, df.iloc[1], df.iloc[2]
        # )
        # action5_reward = self._calc_result_action_reward(
        #     result_target, df.iloc[1], df.iloc[3]
        # )
        # try:
        #     action6_reward = self._calc_result_action_reward(result_target, df.iloc[1], df.iloc[4])
        # except Exception:
        #     action6_reward = 0

        # action7_reward = self._calc_result_action_reward(result_target, df.iloc[2], df.iloc[3])
        # try:
        #     action8_reward = self._calc_result_action_reward(result_target, df.iloc[2], df.iloc[4])
        # except Exception:
        #     action8_reward = 0

        # try:
        #     action9_reward = self._calc_result_action_reward(result_target, df.iloc[3], df.iloc[4])
        # except Exception:
        #     action9_reward = 0

        if action == Action.RANK_ONE_TWO_HORSE:
            reward = action0_reward

        elif action == Action.RANK_ONE_THREE_HORSE:
            reward = action1_reward

        elif action == Action.RANK_TWO_THREE_HORSE:
            reward = action2_reward

        # elif action == Action.RANK_TWO_FIVE_HORSE:
        #     reward = action6_reward

        # elif action == Action.RANK_THREE_FOUR_HORSE:
        #     reward = action7_reward

        # elif action == Action.RANK_THREE_FIVE_HORSE:
        #     reward = action8_reward

        # elif action == Action.RANK_FOUR_FIVE_HORSE:
        #     reward = action9_reward

        else:
            if (
                action0_reward > 0
                or action1_reward > 0
                or action2_reward > 0
                # or action6_reward > 0
                # or action7_reward > 0
                # or action8_reward > 0
                # or action9_reward > 0
            ):
                reward = -5000
            else:
                reward = 10000

        return reward
