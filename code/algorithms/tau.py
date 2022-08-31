"""
Implements the TAU protocol as in the paper.
"""
from __future__ import annotations

from collections import defaultdict
from sys import maxsize
from typing import Literal

from attr import define, field

from negmas.outcomes import Outcome, Issue, OutcomeSpace
from negmas import ensure_os, check_one_and_only
from negmas.gb.constraints import RepeatFinalOfferOnly
from negmas.gb.evaluators import INFINITE, TAUEvaluationStrategy
from negmas.gb.mechanisms import SerialGBMechanism

from negams.gb.common import GBState, ResponseType
from negams.gb.evaluators.base import EvaluationStrategy

__all__ = ["TAUMechanism"]


INFINITE = maxsize
"""Stands for an infinite int value"""

__all__ = ["TAUEvaluationStrategy", "INFINITE"]


@define
class TAUEvaluationStrategy(EvaluationStrategy):
    """
    Implements the Tentative-Accept Unique-Offers Protocol' Evaluation Strategy.
    """

    n_outcomes: int = INFINITE
    cardinality: int = INFINITE
    _accepted: dict[Outcome | None, set[str]] = field(factory=lambda: defaultdict(set))
    _offered: dict[Outcome | None, set[str]] = field(factory=lambda: defaultdict(set))
    _repeating: dict[str, bool] = field(factory=lambda: defaultdict(bool))
    _last: dict[str, Outcome | None] = field(factory=lambda: defaultdict(Outcome))

    def __call__(
        self, negotiator_ids: list[str], state: GBState, history: list[GBState]
    ) -> Outcome | None | Literal["continue"]:
        # keep track of who is repeating. We need to end the negotiation if everyone is repeating
        for source, t in state.threads.items():
            offer = t.new_offer
            self._repeating[source] = self._repeating[source] | (
                offer == self._last[source]
            )
            self._last[source] = offer
        # end the negotiation once all negotiators are repeating
        if (len(self._repeating) == state.n_negotiators) and all(
            list(self._repeating.values())
        ):
            return None

        # it is impossible to have more rounds than the number of outcomes. We should never hit this condition.
        if state.step > self.n_outcomes:
            return None

        # now we can start checking for agreement
        accepted, offered = self._accepted, self._offered

        def register(negotiator, offer, responses):
            """Register the offer and response in offered/accepted dicts"""
            if offer is None:
                return False
            offered[offer].add(negotiator)
            for responder, response in responses.items():
                if response == ResponseType.END_NEGOTIATION:
                    return False
                if response == ResponseType.ACCEPT_OFFER:
                    accepted[offer].add(responder)
            return True

        def registerall(s: GBState):
            """Updates offered/accepted dicts given a state"""
            for source, t in s.threads.items():
                offer, responses = t.new_offer, t.new_responses
                if not register(source, offer, responses):
                    return False
            return True

        # recalcuate accepted, offered if we need to use only a part of the history
        nh, c = len(history), self.cardinality
        if 0 < c <= nh:
            accepted, offered = defaultdict(set), defaultdict(set)
            for s in history[nh - c + 1 :]:
                if not registerall(s):
                    return None

        if not registerall(state):
            return None

        # find outcomes that are accepted and/ or offered
        outcomes = set(accepted.keys()).union(set(offered.keys()))
        n_negotiators = len(negotiator_ids)

        for outcome in outcomes:
            # search for an outcome that was offered and accepted by everyone
            if len(accepted[outcome]) == len(offered[outcome]) == n_negotiators:
                return outcome
        # if we are not ending and found no agreement yet, just continue
        return "continue"

class TAUMechanism(SerialGBMechanism):
    """Implements the TAU protocol using the SerialGBMechanism construct in NegMAS"""
    def __init__(
        self,
        *args,
        cardinality=INFINITE,
        min_unique=0,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | int | None = None,
        **kwargs,
    ):
        check_one_and_only(outcome_space, issues, outcomes)
        outcome_space = ensure_os(outcome_space, issues, outcomes)
        kwargs["evaluator_type"] = TAUEvaluationStrategy
        kwargs["evaluator_params"] = dict(
            cardinality=cardinality, n_outcomes=outcome_space.cardinality
        )
        # implementing the filtering rule
        kwargs["local_constraint_type"] = RepeatFinalOfferOnly
        kwargs["local_constraint_params"] = dict(n=min_unique)
        super().__init__(
            *args,
            outcome_space=outcome_space,
            **kwargs,
        )
