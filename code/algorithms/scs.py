"""
Implements the SCS strategy proposed in the paper.

The implementation is divided into two components: OfferingPolicy and AcceptancePolicy (same as in the paper).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from attr import define, field
from negmas.common import PreferencesChangeType
from negmas.gb.common import ResponseType
from negmas.gb.components.acceptance import SCSAcceptancePolicy
from negmas.gb.components.base import AcceptancePolicy
from negmas.gb.components.base import OfferingPolicy
from negmas.gb.components.offering import SCSOfferingPolicy
from negmas.gb.negotiators.modular.mapneg import MAPNegotiator
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction

if TYPE_CHECKING:
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
__all__ = [
    "SCSNegotiator",
]


@define
class SCSAcceptancePolicy(AcceptancePolicy):
    """
    Implements the Slow Concession Strategy's Acceptance Policy.

    Remarks:
        accepts any offer better than the best I receieved so far as long as it is ratinoal
    """

    def __call__(self, state: GBState, offer: Outcome, source: str) -> ResponseType:
        if not self.negotiator or not self.negotiator.preferences:
            return ResponseType.REJECT_OFFER
        current = state.threads[source].current_offer
        pref = self.negotiator.preferences
        if pref.is_better(offer, current) or (
            current is None and pref.is_not_worse(offer, current)
        ):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class SCSOfferingPolicy(OfferingPolicy):
    """
    Implements the Offering Policy
    """

    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Called to initialize the policy in the start"""
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
        """
        Implements the Offering Policy described in the main paper.

        Remarks:
            - The lexical ordering is implict here as outcomes are always enumerated in the same order in NegMAS
            - The `sorter` is responsible of finding the relative ordering of all outcomes given the ufun
        """
        # Once we start repeating, we keep repeating
        if (
            self._repeating
            or not self.negotiator
            or not self.negotiator.ufun
            or not self.negotiator.nmi
        ):
            return self._last_offer
        # if we have nothing more to offer, repeat
        if self.next_indx >= self.negotiator.nmi.n_outcomes:
            return self._last_offer
        # initalize the sorter if need be
        if not self.sorter:
            self.sorter = PresortingInverseUtilityFunction(self.negotiator.ufun)
            self.sorter.init()
        # Find the next outcome to offer
        outcome = self.sorter.outcome_at(self.next_indx)
        # if there is no next outcome or its utility is worse than reserved, value, repeat
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
        ):
            # self.negotiator.nmi.mechanism.plot()
            # breakpoint()
            self._repeating = True
            return self._last_offer
        # Move to next and save the last outcome
        self.next_indx += 1
        self._last_offer = outcome
        return outcome


class SCSNegotiator(MAPNegotiator):
    """
    Rational Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = SCSAcceptancePolicy()
        kwargs["offering"] = SCSOfferingPolicy()
        super().__init__(*args, **kwargs)
