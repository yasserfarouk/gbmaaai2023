"""
Defines import/export functionality
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from os import PathLike, listdir
from pathlib import Path
from random import random, shuffle
from typing import Callable, Iterable, Sequence

from attr import define

from negmas.outcomes.outcome_space import make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

from .mechanisms import Mechanism
from .negotiators import Negotiator
from .outcomes import (
    CartesianOutcomeSpace,
    DiscreteCartesianOutcomeSpace,
    Issue,
    issues_from_genius,
)
from .preferences import (
    DiscountedUtilityFunction,
    UtilityFunction,
    make_discounted_ufun,
)
from .preferences.value_fun import TableFun

__all__ = [
    "Scenario",
    "load_genius_domain",
    "load_genius_domain_from_folder",
    "find_domain_and_utility_files",
    "get_domain_issues",
]


@define
class Scenario:
    """
    A class representing a negotiation domain
    """

    agenda: CartesianOutcomeSpace
    ufuns: tuple[UtilityFunction, ...]
    mechanism_type: type[Mechanism] | None
    mechanism_params: dict

    @property
    def issues(self) -> tuple[Issue, ...]:
        return self.agenda.issues

    def to_genius_files(self, domain_path: Path, ufun_paths: list[Path]):
        """
        Save domain and ufun files to the `path` as XML.
        """
        domain_path = Path(domain_path)
        ufun_paths = [Path(_) for _ in ufun_paths]
        if len(self.ufuns) != len(ufun_paths):
            raise ValueError(
                f"I have {len(self.ufuns)} ufuns but {len(ufun_paths)} paths were passed!!"
            )
        domain_path.parent.mkdir(parents=True, exist_ok=True)
        self.agenda.to_genius(domain_path)
        for ufun, path in zip(self.ufuns, ufun_paths):
            ufun.to_genius(path, issues=self.issues)
        return self

    def to_genius_folder(self, path: Path):
        """
        Save domain and ufun files to the `path` as XML.
        """
        path.mkdir(parents=True, exist_ok=True)
        domain_name = self.agenda.name.split("/")[-1] if self.agenda.name else "domain"
        ufun_names = [_.name.split("/")[-1] for _ in self.ufuns]
        self.agenda.to_genius(path / domain_name)
        for ufun, name in zip(self.ufuns, ufun_names):
            ufun.to_genius(path / name, issues=self.issues)
        return self

    @property
    def n_negotiators(self) -> int:
        return len(self.ufuns)

    @property
    def n_issues(self) -> int:
        return len(self.agenda.issues)

    @property
    def issue_names(self) -> list[str]:
        return self.agenda.issue_names

    def to_numeric(self) -> Scenario:
        """
        Forces all issues in the domain to become numeric

        Remarks:
            - maps the agenda and ufuns to work correctly together
        """
        raise NotImplementedError()

    def _randomize(self) -> Scenario:
        """
        Randomizes the outcomes in a single issue scneario
        """
        shuffle(self.agenda.issues[0].values)
        return self

    def to_single_issue(
        self, numeric=False, stringify=True, randomize=False
    ) -> Scenario:
        """
        Forces the domain to have a single issue with all possible outcomes

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `DiscreteCategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`
            randomize: Randomize outcome order when creating the single issue

        Remarks:
            - maps the agenda and ufuns to work correctly together
            - Only works if the outcome space is finite
        """
        if hasattr(self.agenda, "issues") and len(self.agenda.issues) == 1:
            return self if not randomize else self._randomize()
        outcomes = list(self.agenda.enumerate_or_sample())
        sos = self.agenda.to_single_issue(numeric, stringify)
        ufuns = []
        souts = list(sos.issues[0].all)
        for u in self.ufuns:
            if isinstance(u, DiscountedUtilityFunction):
                usave = u
                v = u.ufun
                while isinstance(v, DiscountedUtilityFunction):
                    u, v = v, v.ufun
                u.ufun = LinearAdditiveUtilityFunction(
                    values=(TableFun(dict(zip(souts, [v(_) for _ in outcomes]))),),  # type: ignore (The error comes from having LRU cach for table ufun's minmax which should be OK)
                    bias=0.0,
                    reserved_value=v.reserved_value,
                    name=v.name,
                    outcome_space=sos,
                )
                ufuns.append(usave)
                continue
            ufuns.append(
                LinearAdditiveUtilityFunction(
                    values=(TableFun(dict(zip(souts, [u(_) for _ in outcomes]))),),  # type: ignore (The error comes from having LRU cach for table ufun's minmax which should be OK)
                    bias=0.0,
                    reserved_value=u.reserved_value,
                    name=u.name,
                    outcome_space=sos,
                )
            )
        self.ufuns = tuple(ufuns)
        self.agenda = sos
        return self if not randomize else self._randomize()

    def make_session(
        self,
        negotiators: Callable[[], Negotiator] | list[Negotiator] | None = None,
        n_steps: int | float | None = None,
        time_limit: float | None = None,
        roles: list[str] | None = None,
        **kwargs,
    ):
        """
        Generates a ready to run mechanism session for this domain.
        """
        if not self.mechanism_type:
            raise ValueError(
                "Cannot create the domain because it has no `mechanism_type`"
            )

        args = self.mechanism_params
        args.update(kwargs)
        m = self.mechanism_type(
            outcome_space=self.agenda, n_steps=n_steps, time_limit=time_limit, **args
        )
        if not negotiators:
            return m
        negs: list[Negotiator]
        if not isinstance(negotiators, Iterable):
            negs = [negotiators() for _ in range(self.n_negotiators)]
        else:
            negs = list(negotiators)
        if len(self.ufuns) != len(negs) or len(negs) < 1:
            raise ValueError(
                f"Invalid ufuns ({self.ufuns}) or negotiators ({negotiators})"
            )
        if not roles:
            roles = ["negotiator"] * len(negs)
        for n, r, u in zip(negs, roles, self.ufuns):
            m.add(n, preferences=u, role=r)
        return m

    def scale_min(
        self,
        to: float = 1.0,
    ) -> Scenario:
        """Normalizes a utility function to the given range

        Args:
            ufun: The utility function to normalize
            outcomes: A collection of outcomes to normalize for
            rng: range to normalize to. Default is [0, 1]
            levels: Number of levels to use for discretizing continuous issues (if any)
            max_cardinality: Maximum allowed number of outcomes resulting after all discretization is done
        """
        self.ufuns = tuple(_.scale_min(to) for _ in self.ufuns)  # type: ignore The type is correct
        return self

    def scale_max(
        self,
        to: float = 1.0,
    ) -> Scenario:
        """Normalizes a utility function to the given range

        Args:
            ufun: The utility function to normalize
            outcomes: A collection of outcomes to normalize for
            rng: range to normalize to. Default is [0, 1]
            levels: Number of levels to use for discretizing continuous issues (if any)
            max_cardinality: Maximum allowed number of outcomes resulting after all discretization is done
        """
        self.ufuns = tuple(_.scale_max(to) for _ in self.ufuns)  # type: ignore The type is correct
        return self

    def normalize(
        self,
        to: tuple[float, float] = (0.0, 1.0),
    ) -> Scenario:
        """Normalizes a utility function to the given range

        Args:
            rng: range to normalize to. Default is [0, 1]
        """
        self.ufuns = tuple(_.normalize(to) for _ in self.ufuns)  # type: ignore The type is correct
        return self

    def discretize(self, levels: int = 10):
        """Discretize all issues"""
        self.agenda = self.agenda.to_discrete(levels)
        return self

    def remove_discounting(self):
        """Removes discounting from all ufuns"""
        ufuns = []
        for u in self.ufuns:
            while isinstance(u, DiscountedUtilityFunction):
                u = u.ufun
            ufuns.append(u)
        self.ufuns = tuple(ufuns)
        return self

    def remove_reserved_values(self, r: float = float("-inf")):
        """Removes reserved values from all ufuns replaacing it with `r`"""
        for u in self.ufuns:
            u.reserved_value = r
        return self

    @staticmethod
    def from_genius_folder(
        path: PathLike | str,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
    ) -> Scenario | None:
        return load_genius_domain_from_folder(
            folder_name=str(path),
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            safe_parsing=safe_parsing,
        )

    @staticmethod
    def from_genius_files(
        domain: PathLike,
        ufuns: Iterable[PathLike],
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
    ) -> Scenario | None:
        return load_genius_domain(
            domain,
            [_ for _ in ufuns],
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            normalize_utilities=False,
            normalize_max_only=False,
        )


def get_domain_issues(
    domain_file_name: PathLike | str,
    n_discretization: int | None = None,
    safe_parsing=False,
) -> Sequence[Issue] | None:
    """
    Returns the issues of a given XML domain (Genius Format)

    Args:
        domain_file_name: Name of the file
        n_discretization: Number of discrete levels per continuous variable.
        max_cardinality: Maximum number of outcomes in the outcome space after discretization. Used only if `n_discretization` is given.
        safe_parsing: Apply more checks while parsing

    Returns:
        List of issues

    """
    issues = None
    if domain_file_name is not None:
        issues, _ = issues_from_genius(
            domain_file_name,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )
    return issues


def load_genius_domain(
    domain_file_name: PathLike,
    utility_file_names: Iterable[PathLike] | None = None,
    ignore_discount=False,
    ignore_reserved=False,
    safe_parsing=True,
    **kwargs,
) -> Scenario:
    """
    Loads a genius domain, creates appropriate negotiators if necessary

    Args:
        domain_file_name: XML file containing Genius-formatted domain spec
        utility_file_names: XML files containing Genius-fromatted ufun spec
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A `Domain` ready to run

    """
    from .sao import SAOMechanism

    issues = None
    if domain_file_name is not None:
        issues, _ = issues_from_genius(
            domain_file_name,
            safe_parsing=safe_parsing,
        )

    agent_info = []
    if utility_file_names is None:
        utility_file_names = []
    for ufname in utility_file_names:
        utility, discount_factor = UtilityFunction.from_genius(
            file_name=ufname,
            issues=issues,
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            name=str(ufname),
        )
        agent_info.append(
            {
                "ufun": utility,
                "ufun_name": ufname,
                "reserved_value_func": utility.reserved_value
                if utility is not None
                else float("-inf"),
                "discount_factor": discount_factor,
            }
        )
    if domain_file_name is not None:
        kwargs["dynamic_entry"] = False
        kwargs["max_n_agents"] = None
        if not ignore_discount:
            for info in agent_info:
                info["ufun"] = (
                    info["ufun"]
                    if info["discount_factor"] is None or info["discount_factor"] == 1.0
                    else make_discounted_ufun(
                        ufun=info["ufun"],
                        discount_per_round=info["discount_factor"],
                        power_per_round=1.0,
                    )
                )
    if issues is None:
        raise ValueError(f"Could not load domain {domain_file_name}")

    return Scenario(
        agenda=make_os(issues, name=str(domain_file_name)),
        ufuns=[_["ufun"] for _ in agent_info],  # type: ignore
        mechanism_type=SAOMechanism,
        mechanism_params=kwargs,
    )


def load_genius_domain_from_folder(
    folder_name: str | PathLike,
    ignore_reserved=False,
    ignore_discount=False,
    safe_parsing=False,
    **kwargs,
) -> Scenario:
    """
    Loads a genius domain from a folder. See ``load_genius_domain`` for more details.

    Args:
        folder_name: A folder containing one XML domain file and one or more ufun files in Genius format
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A domain ready for `make_session`

    Examples:

        >>> import pkg_resources
        >>> from negmas import load_genius_domain_from_folder
        >>> from negmas import AspirationNegotiator

        Try loading and running a domain with predetermined agents:
        >>> domain = load_genius_domain_from_folder(
        ...     pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop'))
        >>> mechanism = domain.make_session(AspirationNegotiator, n_steps=100)
        >>> state = mechanism.run()
        >>> state.agreement is not None
        True


        Try loading a domain and check the resulting ufuns
        >>> domain = load_genius_domain_from_folder(
        ...     pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop'))

        >>> domain.n_issues, domain.n_negotiators
        (3, 2)

        >>> [type(_) for _ in domain.ufuns]
        [<class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>, <class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>]

        Try loading a domain forcing a single issue space
        >>> domain = load_genius_domain_from_folder(
        ...     pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
        ... ).to_single_issue()
        >>> domain.n_issues, domain.n_negotiators
        (1, 2)
        >>> [type(_) for _ in domain.ufuns]
        [<class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>, <class 'negmas.preferences.crisp.linear.LinearAdditiveUtilityFunction'>]


        Try loading a domain with nonlinear ufuns:
        >>> folder_name = pkg_resources.resource_filename('negmas', resource_name='tests/data/10issues')
        >>> domain = load_genius_domain_from_folder(folder_name)
        >>> print(domain.n_issues)
        10
        >>> print(domain.n_negotiators)
        2
        >>> print([type(u) for u in domain.ufuns])
        [<class 'negmas.preferences.crisp.nonlinear.HyperRectangleUtilityFunction'>, <class 'negmas.preferences.crisp.nonlinear.HyperRectangleUtilityFunction'>]
        >>> u = domain.ufuns[0]
        >>> print(u.outcome_ranges[0])
        {1: (7.0, 9.0), 3: (2.0, 7.0), 5: (0.0, 8.0), 8: (0.0, 7.0)}

        >>> print(u.mappings[0])
        97.0
        >>> print(u([0.0] * domain.n_issues))
        0
        >>> print(u([0.5] * domain.n_issues))
        186.0
    """
    folder_name = str(folder_name)
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = Path(full_name)
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    if domain_file_name is None:
        raise ValueError("Cannot find a domain file")
    return load_genius_domain(
        domain_file_name=domain_file_name,
        utility_file_names=utility_file_names,
        safe_parsing=safe_parsing,
        ignore_reserved=ignore_reserved,
        ignore_discount=ignore_discount,
        **kwargs,
    )


def find_domain_and_utility_files(
    folder_name,
) -> tuple[PathLike | None, list[PathLike]]:
    """Finds the domain and utility_function files in a folder"""
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = full_name
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    return domain_file_name, utility_file_names  # type: ignore
