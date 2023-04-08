import pytest
import pandas as pd

from env.actions import Action
from env.reward import Reward


class TestReward:
    _reward: Reward

    @pytest.fixture(autouse=True)
    def fixture(self):
        self._reward = Reward(
            pd.DataFrame(
                data=[
                    [1, 1, 20030816, 1, 1, 1, 2, 3.000],
                    [2, 2, 20030816, 4, 1, 2, None, 1.200],
                    [3, 3, 20030817, 2, 1, 3, None, 10.400],
                    [4, 1, 20030818, 5, 1, 2, None, 9.100],
                    [5, 1, 20030818, 6, 1, 5, None, 4.500],
                ],
                columns=[
                    "id",
                    "track_id",
                    "date_num",
                    "round",
                    "result_type",
                    "number",
                    "number2",
                    "odds",
                ],
            )
        )

    @pytest.mark.parametrize(
        "action, expected",
        [
            (Action.RANK_ONE_TWO_HORSE, 3000),
            (Action.RANK_ONE_THREE_HORSE, 0),
            (Action.RANK_ONE_FOUR_HORSE, 0),
            (Action.RANK_ONE_FIVE_HORSE, 0),
            (Action.RANK_TWO_THREE_HORSE, 0),
            (Action.RANK_TWO_FOUR_HORSE, 0),
            (Action.NO_ACITON, -5000),
        ],
    )
    def test_action0(self, action: Action, expected):
        actual = self._reward.calc(
            pd.DataFrame(
                data=[
                    [1, 20030816, 1, 1, 0],
                    [1, 20030816, 1, 2, -1],
                    [1, 20030816, 1, 3, -2],
                    [1, 20030816, 1, 4, -3],
                    [1, 20030816, 1, 5, -4],
                    [1, 20030816, 1, 6, -5],
                ],
                columns=["track_id", "date_num", "round", "number", "pred"],
            ),
            action,
        )

        assert actual == expected
