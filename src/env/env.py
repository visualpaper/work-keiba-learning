from dataclasses import dataclass
import gym
import gym.spaces
import numpy as np

from df.keiba_data_frame import KeibaDataFrame
from env.actions import Action
from env.reward import Reward


@dataclass
class ActionCounter:
    _0_count: int = 0
    _1_count: int = 0
    _2_count: int = 0
    _3_count: int = 0
    _total_reward: int = 0

    def add(self, action: int, reward: int):
        
        if action == Action.RANK_ONE_HORSE.value:
            self._0_count += 1
            self._total_reward += reward
        elif action == Action.RANK_TWO_HORSE.value:
            self._1_count += 1
            self._total_reward += reward
        elif action == Action.RANK_THREE_HORSE.value:
            self._2_count += 1
            self._total_reward += reward
        else:
            self._3_count += 1

    def to_string(self) -> str:
        return (
            f"\n"
            f"RANK_ONE_HORSE: {self._0_count}\n"
            f"RANK_TWO_HORSE: {self._1_count}\n"
            f"RANK_THREE_HORSE: {self._2_count}\n"
            f"NO_ACITON: {self._3_count}\n"
            f"TOTAL_REWARD: {self._total_reward}\n"
        )


class KeibaEnv(gym.Env):
    def __init__(self, train_path, result_path):
        super().__init__()
        self._action_counter = ActionCounter()

        # Action 数
        self.action_space = gym.spaces.Discrete(len(Action))
        self.reward_range = (-1, 1)

        # 状態空間
        # 値の定義は明確化せず、Flatten 都合 20 行 51 列の Shape のみ明確化している。
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, 2)
        )

        # Train Data 読み込み準備
        self._train_path = train_path

        # Reward 計算準備
        self._reward = Reward.initialize(result_path)

    def reset(self):
        print(self._action_counter.to_string())
        self._action_counter = ActionCounter()
        self._done = False

        # Train Data 整備
        # ※ 既に AI 側で filter/data 整備は実施済みのため再 type 明確化と、特定列を Drop のみ行っている。
        self._kdf = KeibaDataFrame.from_data(self._train_path)
        self._kdf.setting_types()

        # Train Data 分割
        # ※ 1 レースごとに分割しリストに保持する。
        self._steps = 0
        self._groups = list(self._kdf.groups())
        return self._observe()

    def _observe(self):
        # 1 レースを取得し、observe 化 (20 行固定) する。
        self._race = self._groups[self._steps]
        self._race.reset_index(inplace=True, drop=True)
        #padding_df = self._race.reindex(range(20), fill_value=-1)
        return self._race[["odds", "pred"]].values.tolist()

    def _clipping(self, action, reward):
        if action == Action.NO_ACITON and reward == 10000:
            return 0

        if reward > 0:
            return 1

        return -1

    def step(self, action):

        # Reward 計算
        reward = self._reward.calc(self._race, action)

        self._action_counter.add(action, reward)

        clipping_reward = self._clipping(action, reward)

        # 次の Step に進む
        # ※ steps を +1 した後に状態を取得する必要がある (ただの実装都合...)
        self._steps += 1
        obs = self._observe()
        self._done = self._steps == len(self._groups) - 1

        return obs, clipping_reward, self._done, {}

    def render(self, mode='human'):
        pass
