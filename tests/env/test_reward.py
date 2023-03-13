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
                    [1, 1, 20030816, 1, 1, 1, None, 3.000],
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
            (Action.RANK_ONE_HORSE, 2000),
            (Action.RANK_TWO_HORSE, -1000),
            (Action.RANK_THREE_HORSE, -1000),
            (Action.RANK_FOUR_HORSE, -1000),
            (Action.RANK_FIVE_HORSE, -1000),
            (Action.NO_ACITON, -2500),
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
            action.value,
        )

        assert actual == expected

    @pytest.mark.parametrize(
        "action, expected",
        [
            (Action.RANK_ONE_HORSE, -1000),
            (Action.RANK_TWO_HORSE, 200),
            (Action.RANK_THREE_HORSE, -1000),
            (Action.RANK_FOUR_HORSE, -1000),
            (Action.RANK_FIVE_HORSE, -1000),
            (Action.NO_ACITON, -2500),
        ],
    )
    def test_action1(self, action: Action, expected):
        actual = self._reward.calc(
            pd.DataFrame(
                data=[
                    [2, 20030816, 4, 1, 0],
                    [2, 20030816, 4, 2, -1],
                    [2, 20030816, 4, 3, -2],
                    [2, 20030816, 4, 4, -3],
                    [2, 20030816, 4, 5, -4],
                    [2, 20030816, 4, 6, -5],
                ],
                columns=["track_id", "date_num", "round", "number", "pred"],
            ),
            action.value,
        )

        assert actual == expected

    @pytest.mark.parametrize(
        "action, expected",
        [
            (Action.RANK_ONE_HORSE, -1000),
            (Action.RANK_TWO_HORSE, -1000),
            (Action.RANK_THREE_HORSE, 9400),
            (Action.RANK_FOUR_HORSE, -1000),
            (Action.RANK_FIVE_HORSE, -1000),
            (Action.NO_ACITON, -2500),
        ],
    )
    def test_action2(self, action: Action, expected):
        actual = self._reward.calc(
            pd.DataFrame(
                data=[
                    [3, 20030817, 2, 1, 0],
                    [3, 20030817, 2, 2, -1],
                    [3, 20030817, 2, 3, -2],
                    [3, 20030817, 2, 4, -3],
                    [3, 20030817, 2, 5, -4],
                    [3, 20030817, 2, 6, -5],
                ],
                columns=["track_id", "date_num", "round", "number", "pred"],
            ),
            action.value,
        )

        assert actual == expected

    @pytest.mark.parametrize(
        "action, expected",
        [
            (Action.RANK_ONE_HORSE, -1000),
            (Action.RANK_TWO_HORSE, -1000),
            (Action.RANK_THREE_HORSE, -1000),
            (Action.RANK_FOUR_HORSE, 8100),
            (Action.RANK_FIVE_HORSE, -1000),
            (Action.NO_ACITON, -2500),
        ],
    )
    def test_action3(self, action: Action, expected):
        actual = self._reward.calc(
            pd.DataFrame(
                data=[
                    [1, 20030818, 5, 1, 0],
                    [1, 20030818, 5, 2, -3],
                    [1, 20030818, 5, 3, -1],
                    [1, 20030818, 5, 4, -2],
                    [1, 20030818, 5, 5, -4],
                    [1, 20030818, 5, 6, -5],
                ],
                columns=["track_id", "date_num", "round", "number", "pred"],
            ),
            action.value,
        )

        assert actual == expected

    @pytest.mark.parametrize(
        "action, expected",
        [
            (Action.RANK_ONE_HORSE, -1000),
            (Action.RANK_TWO_HORSE, -1000),
            (Action.RANK_THREE_HORSE, -1000),
            (Action.RANK_FOUR_HORSE, -1000),
            (Action.RANK_FIVE_HORSE, 3500),
            (Action.NO_ACITON, -2500),
        ],
    )
    def test_action4(self, action: Action, expected):
        actual = self._reward.calc(
            pd.DataFrame(
                data=[
                    [1, 20030818, 6, 1, 0],
                    [1, 20030818, 6, 2, -3],
                    [1, 20030818, 6, 3, -1],
                    [1, 20030818, 6, 4, -2],
                    [1, 20030818, 6, 5, -4],
                    [1, 20030818, 6, 6, -5],
                ],
                columns=["track_id", "date_num", "round", "number", "pred"],
            ),
            action.value,
        )

        assert actual == expected

    @pytest.mark.parametrize(
        "action, expected",
        [
            (Action.RANK_ONE_HORSE, -1000),
            (Action.RANK_TWO_HORSE, -1000),
            (Action.RANK_THREE_HORSE, -1000),
            (Action.RANK_FOUR_HORSE, -1000),
            (Action.RANK_FIVE_HORSE, -1000),
            (Action.NO_ACITON, 2500),
        ],
    )
    def test_action5(self, action: Action, expected):
        actual = self._reward.calc(
            pd.DataFrame(
                data=[
                    [2, 20030818, 6, 1, 0],
                    [2, 20030818, 6, 2, -3],
                    [2, 20030818, 6, 3, -1],
                    [2, 20030818, 6, 4, -2],
                    [2, 20030818, 6, 5, -4],
                    [2, 20030818, 6, 6, -5],
                ],
                columns=["track_id", "date_num", "round", "number", "pred"],
            ),
            action.value,
        )

        assert actual == expected
