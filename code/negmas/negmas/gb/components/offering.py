from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from attr import define, field

from negmas import warnings
from negmas.common import PreferencesChangeType, Value
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction

from .base import FilterResult, OfferingPolicy
from .concession import ConcessionRecommender
from .models.ufun import UFunModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.gb.negotiators.base import GBNegotiator
    from negmas.outcomes import Outcome


__all__ = [
    "SCSOfferingPolicy",
    "LimitedOutcomesOfferingPolicy",
    "NegotiatorOfferingPolicy",
    "ConcensusOfferingPolicy",
    "RandomConcensusOfferingPolicy",
    "UnanimousConcensusOfferingPolicy",
    "UtilBasedConcensusOfferingPolicy",
    "MyBestConcensusOfferingPolicy",
    "MyWorstConcensusOfferingPolicy",
    "NoneOfferingPolicy",
    "RandomOfferingPolicy",
    "OfferTop",
    "OfferBest",
    "TFTOfferingPolicy",
    "MiCROOfferingPolicy",
]


@define
class MiCROOfferingPolicy(OfferingPolicy):
    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _received: set[Outcome] = field(factory=set)
    _sent: set[Outcome] = field(factory=set)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scaled,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            self.sorter = PresortingInverseUtilityFunction(self.negotiator.ufun)
            self.sorter.init()
            self.next_indx = 0
            self._received = set()
            self._sent = set()

    def sample_sent(self) -> Outcome | None:
        if not len(self._sent):
            return None
        return random.choice(list(self._sent))

    def ensure_sorter(self):
        if not self.sorter:
            assert self.negotiator.ufun
            self.sorter = PresortingInverseUtilityFunction(self.negotiator.ufun)
            self.sorter.init()
        return self.sorter

    def next_offer(self) -> Outcome | None:
        return self.ensure_sorter().outcome_at(self.next_indx)

    def best_offer_so_far(self) -> Outcome | None:
        if self.next_indx > 0:
            return self.ensure_sorter().outcome_at(self.next_indx - 1)
        return None

    def ready_to_concede(self) -> bool:
        return len(self._sent) <= len(self._received)

    def __call__(self, state: GBState) -> Outcome | None:
        outcome = self.next_offer()
        assert self.sorter
        assert self.negotiator.ufun
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
            or not self.ready_to_concede()
        ):
            return self.sample_sent()
        self.next_indx += 1
        self._sent.add(outcome)
        return outcome

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        self._received.add(offer)
        return super().on_partner_proposal(state, partner_id, offer)


@define
class SCSOfferingPolicy(OfferingPolicy):

    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scaled,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            self.sorter = PresortingInverseUtilityFunction(self.negotiator.ufun)
            self.sorter.init()
            self.next_indx = 0
            self._repeating = False

    def __call__(self, state: GBState) -> Outcome | None:
        if (
            self._repeating
            or not self.negotiator
            or not self.negotiator.ufun
            or not self.negotiator.nmi
        ):
            return self._last_offer
        if self.next_indx >= self.negotiator.nmi.n_outcomes:
            return self._last_offer
        if not self.sorter:
            self.sorter = PresortingInverseUtilityFunction(self.negotiator.ufun)
            self.sorter.init()
        outcome = self.sorter.outcome_at(self.next_indx)
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
        ):
            # self.negotiator.nmi.mechanism.plot()
            # breakpoint()
            self._repeating = True
            return self._last_offer
        self.next_indx += 1
        self._last_offer = outcome
        return outcome


@define
class TFTOfferingPolicy(OfferingPolicy):
    """
    An acceptance strategy that concedes as much as the partner (or more)
    """

    partner_ufun: UFunModel
    recommender: ConcessionRecommender
    stochastic: bool = False
    _partner_offer: Outcome | None = field(init=False, default=None)

    def before_responding(self, state: GBState, offer: Outcome | None, source: str):
        self._partner_offer = offer

    def __call__(self, state):
        if not self.negotiator or not self.negotiator.ufun:
            return None
        partner_u = (
            float(self.partner_ufun.eval_normalized(self._partner_offer, True))
            if self._partner_offer
            else 1.0
        )
        partner_concession = 1.0 - partner_u
        my_concession = self.recommender(partner_concession, state)
        if not math.isfinite(my_concession):
            warnings.warn(
                f"Got {my_concession} for concession which is unacceptable. Will use no concession"
            )
            my_concession = 0.0
        if not (-1e-6 <= my_concession <= 1.0000001):
            warnings.warn(f"{my_concession} is negative or above 1")
            my_concession = 0.0
        target_utility = 1.0 - float(my_concession)
        if self.stochastic:
            return self.negotiator.ufun.invert().one_in(
                (target_utility, 1.0), normalized=True
            )
        return self.negotiator.ufun.invert().worst_in(
            (target_utility, 1.0), normalized=True
        )


@define
class OfferBest(OfferingPolicy):
    """
    Offers Only the best outcome.

    Remarks:
        - You can pass the  best outcome if you know it as `best` otherwise it will find it.
    """

    _best: Outcome | None = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        _, self._best = self.negotiator.ufun.extreme_outcomes()

    def __call__(self, state: GBState) -> Outcome | None:
        return self._best


@define
class OfferTop(OfferingPolicy):
    """
    Offers outcomes that are in the given top fraction or top `k`. If neither is given it reverts to only offering the best outcome

    Remarks:
        - The outcome-space is always discretized and the constraints `fraction` and `k` are applied to the discretized space
    """

    fraction: float = 0.0
    k: int = 1

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        if any(
            _.type
            not in (
                PreferencesChangeType.Scaled,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            self.negotiator.ufun.invert().init()

    def __call__(self, state: GBState) -> Outcome | None:
        if not self.negotiator or not self.negotiator.ufun:
            return None
        top_k = self.negotiator.ufun.invert().within_indices((0, self.k))
        top_f = self.negotiator.ufun.invert().within_fractions((0.0, self.fraction))
        top = list(set(top_k + top_f))
        return random.choice(top)


@define
class NoneOfferingPolicy(OfferingPolicy):
    """
    Always offers `None` which means it never gets an agreement.
    """

    def __call__(self, state: GBState) -> Outcome | None:
        return None


@define
class RandomOfferingPolicy(OfferingPolicy):
    """
    Always offers `None` which means it never gets an agreement.
    """

    def __call__(self, state: GBState) -> Outcome | None:
        if not self.negotiator or not self.negotiator.nmi:
            return None
        return self.negotiator.nmi.random_outcomes(1)[0]


@define
class LimitedOutcomesOfferingPolicy(OfferingPolicy):
    """
    Offers from a given list of outcomes
    """

    outcomes: list[Outcome] | None
    prob: list[float] | None = None
    p_ending: float = 0.0

    def __call__(self, state: GBState, retry=False) -> Outcome | None:
        if not self.negotiator or not self.negotiator.nmi:
            return None
        if random.random() < self.p_ending - 1e-7:
            return None
        if not self.prob or not self.outcomes:
            return random.choice(
                self.outcomes
                if self.outcomes
                else list(self.negotiator.nmi.discrete_outcomes())
            )
        r, s = random.random(), 0.0
        for w, p in zip(self.outcomes, self.prob):
            s += p
            if r <= s:
                return w
        if retry:
            return None
        if s > 0.999:
            return self.outcomes[-1]
        self.prob = [_ / s for _ in self.prob]
        return self(state, True)


@define
class NegotiatorOfferingPolicy(OfferingPolicy):
    """
    Uses a negotiator as an offering strategy
    """

    proposer: GBNegotiator = field(kw_only=True)

    def __call__(self, state: GBState) -> Outcome | None:
        return self.proposer.propose(state)


@define
class ConcensusOfferingPolicy(OfferingPolicy, ABC):
    """
    Offers based on concensus of multiple strategies
    """

    strategies: list[OfferingPolicy]

    def filter(self, indx: int, offer: Outcome | None) -> FilterResult:
        """
        Called with the decision of each strategy in order.


        Remarks:
            - Two decisions need to be made:

              1. Should we continue trying other strategies
              2. Should we save this result.
        """
        return FilterResult(True, True)

    @abstractmethod
    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        """
        Called to make a final decsision given the decisions of the stratgeis with indices `indices` (see `filter` for filtering rules)
        """

    def __call__(self, state: GBState) -> Outcome | None:
        selected, selected_indices = [], []
        for i, s in enumerate(self.strategies):
            response = s.propose(state)
            r = self.filter(i, response)
            if not r.next:
                break
            if r.save:
                selected.append(response)
                selected_indices.append(i)

        return self.decide(selected_indices, selected)


@define
class UnanimousConcensusOfferingPolicy(ConcensusOfferingPolicy):
    """
    Offers only if all offering strategies gave exactly the same outcome
    """

    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        outcomes = set(responses)
        if len(outcomes) != 1:
            return None
        return list(outcomes)[0]


@define
class RandomConcensusOfferingPolicy(ConcensusOfferingPolicy):
    """
    Offers a random response from the list of stratgies (different strategy every time).
    """

    prob: list[float] | None = None

    def __attrs_post_init__(self):
        if not self.prob:
            return
        s = sum(self.prob)
        self.prob = [_ / s for _ in self.prob]

    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        if not self.prob:
            return random.choice(responses)

        r, s = random.random(), 0.0
        for i, p in enumerate(self.prob):
            s += p
            if r <= s:
                return responses[i]
        if s > 0.999:
            return responses[-1]
        raise ValueError(f"sum of probabilities is less than 1: {s}")


@define
class UtilBasedConcensusOfferingPolicy(ConcensusOfferingPolicy, ABC):
    """
    Offers from the list of stratgies (different strategy every time) based on outcome utilities
    """

    @abstractmethod
    def decide_util(self, utils: list[Value]) -> int:
        """
        Returns the index to chose based on utils
        """

    def decide(
        self, indices: list[int], responses: list[Outcome | None]
    ) -> Outcome | None:
        if not self.negotiator.ufun:
            raise ValueError(f"Cannot decide because I have no ufun")
        return responses[
            self.decide_util([self.negotiator.ufun(_) for _ in set(responses)])
        ]


@define
class MyBestConcensusOfferingPolicy(UtilBasedConcensusOfferingPolicy):
    """
    Offers my best outcome from the list of stratgies (different strategy every time).
    """

    def decide_util(self, utils: list[Value]) -> int:
        return max(range(len(utils)), key=lambda x: utils[x])


@define
class MyWorstConcensusOfferingPolicy(UtilBasedConcensusOfferingPolicy):
    """
    Offers my worst outcome from the list of stratgies (different strategy every time) based on outcome utilities
    """

    def decide_util(self, utils: list[Value]) -> int:
        return min(range(len(utils)), key=lambda x: utils[x])
