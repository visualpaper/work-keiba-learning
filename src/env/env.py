import gym
import gym.spaces
import numpy as np

from df.keiba_data_frame import KeibaDataFrame
from env.actions import Action
from env.result_summary import ResultSummary
from env.reward import Reward


class KeibaEnv(gym.Env):
    def __init__(self, train_path, result_path):
        super().__init__()
        self._result_summary = ResultSummary.initialize(Action.values())

        # Action 数
        self.action_space = gym.spaces.Discrete(len(Action))
        self.reward_range = (-1, 1)

        # 状態空間
        # 値の定義は明確化せず、Flatten 都合 20 行 51 列の Shape のみ明確化している。
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20, 2))

        # Train Data 読み込み準備
        self._train_path = train_path

        # Reward 計算準備
        self._reward = Reward.initialize(result_path)

    def reset(self):
        print(self._result_summary.to_result())

        self._result_summary = ResultSummary.initialize(Action.values())
        self._done = False

        # Train Data 整備
        # ※ 既に AI 側で filter/data 整備は実施済みのため再 type 明確化と、特定列を Drop のみ行っている。
        self._kdf = KeibaDataFrame.from_datas(self._train_path)
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
        # padding_df = self._race.reindex(range(5), fill_value=-1)
        padding_df = self._race[["odds", "pred"]].reindex(range(20), fill_value=-1)
        # print(padding_df.values.tolist())
        return padding_df.values.tolist()

    def _clipping(self, action: Action, reward: int):

        if reward > 0:
            return 1

        return -1

    def step(self, action):
        step_action = Action.of(int(action))

        # Reward 計算
        reward = self._reward.calc(self._race, step_action)

        self._result_summary.add(step_action, reward)

        clipping_reward = self._clipping(step_action, reward)

        # 次の Step に進む
        # ※ steps を +1 した後に状態を取得する必要がある (ただの実装都合...)
        self._steps += 1
        obs = self._observe()
        self._done = self._steps == len(self._groups) - 1

        return obs, clipping_reward, self._done, {}

    def render(self, mode='human'):
        pass
