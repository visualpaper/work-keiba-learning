from typing import List
import pytest

from env.actions import Action
from env.result_summary import ResultSummary


class TestRewardSummary:
    @pytest.mark.parametrize(
        "actions, expected",
        [
            (
                [Action.RANK_ONE_TWO_HORSE],
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        }
                    }
                ),
            ),
            (
                [Action.RANK_ONE_TWO_HORSE, Action.NO_ACITON],
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
            ),
        ],
    )
    def test_initialize(self, actions: List[Action], expected: ResultSummary):
        actual = ResultSummary.initialize(actions)

        assert actual == expected

    @pytest.mark.parametrize(
        "summary, action, reward, expected",
        [
            (
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
                Action.RANK_ONE_TWO_HORSE,
                0,
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 1,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
            ),
            (
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
                Action.RANK_ONE_TWO_HORSE,
                3200,
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 1,
                            'hazure': 0,
                            'reward': 3200,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
            ),
            (
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
                Action.NO_ACITON,
                10000,
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 1, 'hazure': 0, 'reward': 0},
                    }
                ),
            ),
            (
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 0, 'reward': 0},
                    }
                ),
                Action.NO_ACITON,
                -5000,
                ResultSummary(
                    {
                        Action.RANK_ONE_TWO_HORSE.value: {
                            'atari': 0,
                            'hazure': 0,
                            'reward': 0,
                        },
                        Action.NO_ACITON.value: {'atari': 0, 'hazure': 1, 'reward': 0},
                    }
                ),
            ),
        ],
    )
    def test_add(
        self,
        summary: ResultSummary,
        action: Action,
        reward: int,
        expected: ResultSummary,
    ):
        summary.add(action, reward)

        print(summary.to_result())
        assert summary.to_result() == expected.to_result()
