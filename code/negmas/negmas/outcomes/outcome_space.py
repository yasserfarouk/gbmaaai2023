from __future__ import annotations

import random
from functools import reduce
from itertools import filterfalse
from operator import mul
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, Union

from attr import define, field

from negmas.helpers import unique_name
from negmas.helpers.types import get_full_type_name
from negmas.outcomes.outcome_ops import (
    cast_value_types,
    outcome_is_valid,
    outcome_types_are_ok,
)
from negmas.protocols import XmlSerializable
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize
from negmas.warnings import NegmasSpeedWarning, warn

from .base_issue import DiscreteIssue, Issue
from .categorical_issue import CategoricalIssue
from .common import Outcome
from .contiguous_issue import ContiguousIssue
from .issue_ops import (
    enumerate_discrete_issues,
    issues_from_outcomes,
    issues_from_xml_str,
    issues_to_xml_str,
    sample_issues,
)
from .protocols import DiscreteOutcomeSpace, OutcomeSpace
from .range_issue import RangeIssue

if TYPE_CHECKING:
    from negmas.preferences.protocols import HasReservedOutcome, HasReservedValue

__all__ = [
    "CartesianOutcomeSpace",
    "DiscreteCartesianOutcomeSpace",
    "make_os",
    "DistanceFun",
]

NLEVELS = 5


DistanceFun = Callable[[Outcome, Outcome, Union[OutcomeSpace, None]], float]
"""A callable that can calculate the distance between two outcomes in an outcome-space"""


def make_os(
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    name: str | None = None,
) -> CartesianOutcomeSpace:
    """
    A factory to create outcome-spaces from lists of `Issue` s or `Outcome` s.

    Remarks:

        - must pass one and exactly one of `issues` and `outcomes`
    """
    if issues and outcomes:
        raise ValueError(
            f"Cannot make an outcome space passing both issues and outcomes"
        )
    if not issues and not outcomes:
        raise ValueError(
            f"Cannot make an outcome space without passing issues or outcomes"
        )
    if not issues and outcomes:
        issues_ = issues_from_outcomes(outcomes)
    else:
        issues_ = issues
    if issues_ is None:
        raise ValueError(
            f"Cannot make an outcome space without passing issues or outcomes"
        )

    issues_ = tuple(issues_)
    if all(_.is_discrete() for _ in issues_):
        return DiscreteCartesianOutcomeSpace(issues_, name=name if name else "")
    return CartesianOutcomeSpace(issues_, name=name if name else "")


@define(frozen=True)
class CartesianOutcomeSpace(XmlSerializable):
    """
    An outcome-space that is generated by the cartesian product of a tuple of `Issue` s.
    """

    issues: tuple[Issue, ...] = field(converter=tuple)
    name: str | None = field(eq=False, default=None)

    def __attrs_post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", unique_name("os", add_time=False, sep=""))

    def contains_issue(self, x: Issue) -> bool:
        """Cheks that the given issue is in the tuple of issues constituting the outcome space (i.e. it is one of its dimensions)"""
        return x in self.issues

    def is_valid(self, outcome: Outcome) -> bool:
        return outcome_is_valid(outcome, self.issues)

    def is_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return all(_.is_discrete() for _ in self.issues)

    def is_finite(self) -> bool:
        """Checks whether the space is finite"""
        return self.is_discrete()

    def contains_os(self, x: OutcomeSpace) -> bool:
        """Checks whether an outcome-space is contained in this outcome-space"""
        if isinstance(x, CartesianOutcomeSpace):
            return len(self.issues) == len(x.issues) and all(
                b in a for a, b in zip(self.issues, x.issues)
            )
        if self.is_finite() and not x.is_finite():
            return False
        if not self.is_finite() and not x.is_finite():
            raise NotImplementedError(
                "Cannot check an infinite outcome space that is not cartesian for inclusion in an infinite cartesian outcome space!!"
            )
        warn(
            f"Testing inclusion of a finite non-carteisan outcome space in a cartesian outcome space can be very slow (will do {x.cardinality} checks)",
            NegmasSpeedWarning,
        )
        return all(self.is_valid(_) for _ in x.enumerate())  # type: ignore If we are here, we know that x is finite

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            **d,
            name=self.name,
            issues=serialize(self.issues),
        )

    @classmethod
    def from_dict(cls, d):
        return cls(**deserialize(d))  # type: ignore

    @property
    def issue_names(self) -> list[str]:
        """Returns an ordered list of issue names"""
        return [_.name for _ in self.issues]

    @property
    def cardinality(self) -> int | float:
        """The space cardinality = the number of outcomes"""
        return reduce(mul, [_.cardinality for _ in self.issues], 1)

    def is_compact(self) -> bool:
        """Checks whether all issues are complete ranges"""
        return all(isinstance(_, RangeIssue) for _ in self.issues)

    def is_all_continuous(self) -> bool:
        """Checks whether all issues are discrete"""
        return all(_.is_continuous() for _ in self.issues)

    def is_not_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return any(_.is_continuous() for _ in self.issues)

    def is_numeric(self) -> bool:
        """Checks whether all issues are numeric"""
        return all(_.is_numeric() for _ in self.issues)

    def is_integer(self) -> bool:
        """Checks whether all issues are integer"""
        return all(_.is_integer() for _ in self.issues)

    def is_float(self) -> bool:
        """Checks whether all issues are real"""
        return all(_.is_float() for _ in self.issues)

    def to_discrete(
        self, levels: int | float = 10, max_cardinality: int | float = float("inf")
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Discretizes the outcome space by sampling `levels` values for each continuous issue.

        The result of the discretization is stable in the sense that repeated calls will return the same output.
        """
        if max_cardinality != float("inf"):
            c = reduce(
                mul,
                [_.cardinality if _.is_discrete() else levels for _ in self.issues],
                1,
            )
            if c > max_cardinality:
                raise ValueError(
                    f"Cannot convert OutcomeSpace to a discrete OutcomeSpace with at most {max_cardinality} (at least {c} outcomes are required)"
                )
        issues = tuple(
            issue.to_discrete(
                levels if issue.is_continuous() else None,
                compact=False,
                grid=True,
                endpoints=True,
            )
            for issue in self.issues
        )
        return DiscreteCartesianOutcomeSpace(issues=issues, name=self.name)

    @classmethod
    def from_xml_str(
        cls, xml_str: str, safe_parsing=True, name=None
    ) -> CartesianOutcomeSpace:
        issues, _ = issues_from_xml_str(
            xml_str,
            safe_parsing=safe_parsing,
            n_discretization=None,
        )
        if not issues:
            raise ValueError(f"Failed to read an issue space from an xml string")
        issues = tuple(issues)
        if all(isinstance(_, DiscreteIssue) for _ in issues):
            return DiscreteCartesianOutcomeSpace(issues, name=name)
        return cls(issues, name=name)

    @staticmethod
    def from_outcomes(
        outcomes: list[Outcome],
        numeric_as_ranges: bool = False,
        issue_names: list[str] | None = None,
        name: str | None = None,
    ) -> DiscreteCartesianOutcomeSpace:
        return DiscreteCartesianOutcomeSpace(
            issues_from_outcomes(outcomes, numeric_as_ranges, issue_names), name=name
        )

    def to_xml_str(self) -> str:
        return issues_to_xml_str(self.issues)

    def are_types_ok(self, outcome: Outcome) -> bool:
        """Checks if the type of each value in the outcome is correct for the given issue"""
        return outcome_types_are_ok(outcome, self.issues)

    def ensure_correct_types(self, outcome: Outcome) -> Outcome:
        """Returns an outcome that is guaratneed to have correct types or raises an exception"""
        return cast_value_types(outcome, self.issues)

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = True,
        fail_if_not_enough=True,
    ) -> Iterable[Outcome]:
        return sample_issues(
            self.issues, n_outcomes, with_replacement, fail_if_not_enough
        )

    def cardinality_if_discretized(
        self, levels: int, max_cardinality: int | float = float("inf")
    ) -> int:
        c = reduce(
            mul,
            [_.cardinality if _.is_discrete() else levels for _ in self.issues],
            1,
        )
        return min(c, max_cardinality)

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteCartesianOutcomeSpace:
        for l in range(levels, 0, -1):
            if self.cardinality_if_discretized(levels) < max_cardinality:
                break
        else:
            raise ValueError(
                f"Cannot discretize with levels <= {levels} keeping the cardinality under {max_cardinality} Outocme space cardinality is {self.cardinality}\nOutcome space: {self}"
            )
        return self.to_discrete(l, max_cardinality, **kwargs)

    def enumerate_or_sample_rational(
        self,
        preferences: Iterable[HasReservedValue | HasReservedOutcome],
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
        aggregator: Callable[[Iterable[bool]], bool] = any,
    ) -> Iterable[Outcome]:
        """
        Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise.

        Args:
            preferences: A list of `Preferences` that is used to judge outcomes
            levels: The number of levels to use for discretization if needed
            max_cardinality: The maximum cardinality allowed in case of discretization
            aggregator: A predicate that takes an `Iterable` of booleans representing whether or not an outcome is rational
                        for a given `Preferences` (i.e. better than reservation) and returns a single boolean representing
                        the result for all preferences. Default is any but can be all.
        """
        from negmas.preferences.protocols import HasReservedOutcome, HasReservedValue

        if (
            levels == float("inf")
            and max_cardinality == float("inf")
            and not self.is_discrete()
        ):
            raise ValueError(
                "Cannot enumerate-or-sample an outcome space with infinite outcomes without specifying `levels` and/or `max_cardinality`"
            )
        from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace

        if isinstance(self, DiscreteCartesianOutcomeSpace):
            results = self.enumerate()  # type: ignore We know the outcome space is correct
        else:
            if max_cardinality == float("inf"):
                return self.to_discrete(
                    levels=levels, max_cardinality=max_cardinality
                ).enumerate()
            results = self.sample(
                int(max_cardinality), with_replacement=False, fail_if_not_enough=False
            )

        def is_irrational(x: Outcome):
            def irrational(u: HasReservedOutcome | HasReservedValue, x: Outcome):
                if isinstance(u, HasReservedValue):
                    if u.reserved_value is None:
                        return False
                    return u(x) < u.reserved_value  # type: ignore
                if isinstance(u, HasReservedOutcome) and u.reserved_outcome is not None:
                    return u.is_worse(x, u.reserved_outcome)  # type: ignore
                return False

            return aggregator(irrational(u, x) for u in preferences)

        return filterfalse(lambda x: is_irrational(x), results)

    def enumerate_or_sample(
        self,
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
    ) -> Iterable[Outcome]:
        """Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise"""
        if (
            levels == float("inf")
            and max_cardinality == float("inf")
            and not self.is_discrete()
        ):
            raise ValueError(
                "Cannot enumerate-or-sample an outcome space with infinite outcomes without specifying `levels` and/or `max_cardinality`"
            )
        from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace

        if isinstance(self, DiscreteCartesianOutcomeSpace):
            return self.enumerate()  # type: ignore We know the outcome space is correct
        if max_cardinality == float("inf"):
            return self.to_discrete(
                levels=levels, max_cardinality=max_cardinality
            ).enumerate()
        return self.sample(
            int(max_cardinality), with_replacement=False, fail_if_not_enough=False
        )

    def to_single_issue(
        self,
        numeric=False,
        stringify=True,
        levels: int = NLEVELS,
        max_cardinality: int | float = float("inf"),
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Creates a new outcome space that is a single-issue version of this one discretizing it as needed

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otberwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`
            levels: Number of levels to discretize any continuous issue
            max_cardinality: Maximum allowed number of outcomes in the resulting issue.

        Remarks:
            - Will discretize inifinte outcome spaces
        """
        if isinstance(self, DiscreteCartesianOutcomeSpace) and len(self.issues) == 1:
            return self
        dos = self.to_discrete(levels, max_cardinality)
        return dos.to_single_issue(numeric, stringify)  # type: ignore

    def __contains__(self, item):
        if isinstance(item, OutcomeSpace):
            return self.contains_os(item)
        if isinstance(item, Issue):
            return self.contains_issue(item)
        if isinstance(item, Outcome):
            return self.is_valid(item)
        if not isinstance(item, Sequence):
            return False
        if not item:
            return True
        if isinstance(item[0], Issue):
            return len(self.issues) == len(item) and self.contains_os(
                make_os(issues=item)
            )
        if isinstance(item[0], Outcome):
            return len(self.issues) == len(item) and self.contains_os(
                make_os(outcomes=item)
            )
        return False


@define(frozen=True)
class DiscreteCartesianOutcomeSpace(CartesianOutcomeSpace):
    """
    A discrete outcome-space that is generated by the cartesian product of a tuple of `Issue` s (i.e. with finite number of outcomes).
    """

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteCartesianOutcomeSpace:
        return self

    def __attrs_post_init__(self):
        for issue in self.issues:
            if not issue.is_discrete():
                raise ValueError(
                    f"Issue is not discrete. Cannot be added to a DiscreteOutcomeSpace. You must discretize it first: {issue} "
                )

    @property
    def cardinality(self) -> int:
        return reduce(mul, [_.cardinality for _ in self.issues], 1)

    def cardinality_if_discretized(
        self, levels: int, max_cardinality: int | float = float("inf")
    ) -> int:
        return self.cardinality

    def enumerate(self) -> Iterable[Outcome]:
        return enumerate_discrete_issues(
            self.issues  #  type: ignore I know that all my issues are actually discrete
        )

    def limit_cardinality(
        self,
        max_cardinality: int | float = float("inf"),
        levels: int | float = float("inf"),
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)

        Args:
            max_cardinality: The maximum number of outcomes in the resulting space
            levels: The maximum number of levels for each issue/subissue
        """
        if self.cardinality <= max_cardinality or all(
            _.cardinality < levels for _ in self.issues
        ):
            return self
        new_levels = [_.cardinality for _ in self.issues]  # type: ignore will be corrected the next line
        new_levels = [int(_) if _ < levels else int(levels) for _ in new_levels]
        new_cardinality = reduce(mul, new_levels, 1)

        def _reduce_total_cardinality(new_levels, max_cardinality, new_cardinality):
            sort = reversed(sorted((_, i) for i, _ in enumerate(new_levels)))
            sorted_levels = [_[0] for _ in sort]
            indices = [_[1] for _ in sort]
            needed = new_cardinality - max_cardinality
            current = 0
            n = len(sorted_levels)
            while needed > 0 and current < n:
                nxt = n - 1
                v = sorted_levels[current]
                if v == 1:
                    continue
                for i in range(current + 1, n - 1):
                    if v == sorted_levels[i]:
                        continue
                    nxt = i
                    break
                diff = v - sorted_levels[nxt]
                if not diff:
                    diff = 1
                new_levels[indices[current]] -= 1
                max_cardinality = (max_cardinality // v) * (v - 1)
                sort = reversed(sorted((_, i) for i, _ in enumerate(new_levels)))
                sorted_levels = [_[0] for _ in sort]
                current = 0
                needed = new_cardinality - max_cardinality
            return new_levels

        if new_cardinality > max_cardinality:
            new_levels: list[int] = _reduce_total_cardinality(
                new_levels, max_cardinality, new_cardinality
            )
        issues: list[Issue] = []
        for j, i, issue in zip(
            new_levels, (_.cardinality for _ in self.issues), self.issues
        ):
            issues.append(issue if j >= i else issue.to_discrete(j, compact=True))
        return DiscreteCartesianOutcomeSpace(
            tuple(issues), name=f"{self.name}-{max_cardinality}"
        )

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        return True

    def to_discrete(self, *args, **kwargs) -> DiscreteOutcomeSpace:
        return self

    def to_single_issue(
        self, numeric=False, stringify=True
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Creates a new outcome space that is a single-issue version of this one

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`

        Remarks:
            - maps the agenda and ufuns to work correctly together
            - Only works if the outcome space is finite
        """
        outcomes = list(self.enumerate())
        values = (
            range(len(outcomes))
            if numeric
            else [f"v{_}" for _ in range(len(outcomes))]
            if stringify
            else outcomes
        )
        issue = (
            ContiguousIssue(len(outcomes), name="-".join(self.issue_names))
            if numeric
            else CategoricalIssue(values, name="-".join(self.issue_names))
        )
        return DiscreteCartesianOutcomeSpace(
            issues=(issue,),
            name=self.name,
        )

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = False,
        fail_if_not_enough=True,
    ) -> Iterable[Outcome]:
        """
        Samples up to n_outcomes with or without replacement.

        This methor provides a base implementation that is not memory efficient.
        It will simply create a list of all outcomes using `enumerate()` and then
        samples from it. Specific outcome space types should override this method
        to improve its efficiency if possible.

        """
        outcomes = self.enumerate()
        outcomes = list(outcomes)
        if with_replacement:
            return random.choices(outcomes, k=n_outcomes)
        if fail_if_not_enough and n_outcomes > self.cardinality:
            raise ValueError("Cannot sample enough")
        random.shuffle(outcomes)
        return outcomes[:n_outcomes]

    def __iter__(self):
        return self.enumerate().__iter__()

    def __len__(self) -> int:
        return self.cardinality