from dataclasses import dataclass
from typing import List

from env.actions import Action


@dataclass(frozen=True)
class ResultSummary:
    RESULT_FORMAT = "{action}: {reward_summary} {tekityu} {kaisyu} ({atari}/{total})"

    # atari: 当たり馬券数
    # hazure: はずれ馬券数
    # reward: 購入金額を引いていない当たった金額
    _summary: dict

    @classmethod
    def initialize(cls, actions: List[Action]):

        return cls(
            {action.value: {"atari": 0, "hazure": 0, "reward": 0} for action in actions}
        )

    def add(self, action: Action, reward: int) -> None:
        target = self._summary[action.value]

        if reward > 0:
            target["atari"] += 1
            if action != Action.NO_ACITON:
                target["reward"] += reward
        else:
            target["hazure"] += 1

        self._summary[action.value] = target

    def to_result(self) -> str:
        result = [
            self._to_one_result(Action.of(key), value)
            for key, value in self._summary.items()
        ]
        result.append(self._to_total_result())

        return "\n" + "\n".join(result)

    def _calc_kounyu_amount_total(self, action: Action, kounyu_count_total: int) -> int:

        if action == Action.NO_ACITON:
            return 0

        return kounyu_count_total * 1000

    def _to_result(
        self,
        caption: str,
        atari_count_total: int,
        reward_total: int,
        kounyu_count_total: int,
        kounyu_amount_total: int,
    ) -> str:

        # 儲け合計
        reward_summary = reward_total - kounyu_amount_total

        # 的中率
        try:
            tekityu = "的中率 {:.2%}".format(atari_count_total / kounyu_count_total)
        except Exception:
            tekityu = "的中率 {:.2%}".format(0)

        # 回収率
        try:
            kaisyu = "回収率 {:.2%}".format(reward_total / kounyu_amount_total)
        except Exception:
            kaisyu = "回収率 {:.2%}".format(0)

        return self.RESULT_FORMAT.format(
            action=caption.ljust(30),
            # 儲け合計
            reward_summary=(str(reward_summary) + "円").rjust(15),
            # 的中率
            tekityu=tekityu.ljust(12),
            # 回収率
            kaisyu=kaisyu.ljust(12),
            # 当たり数合計
            atari=atari_count_total,
            # 購入数合計
            total=kounyu_count_total,
        )

    def _to_one_result(self, key: Action, value: dict) -> str:
        atari_count_total = value["atari"]
        hazure_count_total = value["hazure"]
        reward_total = value["reward"]

        # 購入数合計
        kounyu_count_total = atari_count_total + hazure_count_total

        # 購入金額合計
        kounyu_amount_total = self._calc_kounyu_amount_total(key, kounyu_count_total)

        return self._to_result(
            key.name,
            atari_count_total,
            reward_total,
            kounyu_count_total,
            kounyu_amount_total,
        )

    def _to_total_result(self) -> str:
        atari_count_total = 0
        hazure_count_total = 0
        reward_total = 0
        kounyu_count_total = 0
        kounyu_amount_total = 0

        for key, value in self._summary.items():
            action = Action.of(key)
            if action == Action.NO_ACITON:
                continue

            atari_count_total += value["atari"]
            hazure_count_total += value["hazure"]
            reward_total += value["reward"]

            # 購入数合計
            kounyu_count_total += value["atari"] + value["hazure"]

            # 購入金額合計
            kounyu_amount_total += self._calc_kounyu_amount_total(
                action, value["atari"] + value["hazure"]
            )

        return self._to_result(
            "TOTAL",
            atari_count_total,
            reward_total,
            kounyu_count_total,
            kounyu_amount_total,
        )
