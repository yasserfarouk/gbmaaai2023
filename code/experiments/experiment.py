from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from multiprocessing import cpu_count
from os import unlink
from pathlib import Path
from random import shuffle
from socket import gethostname
import sys
from time import perf_counter
from typing import Any, Type

from attr import dataclass
from negmas.gb import SCSNegotiator, TAUMechanism
from negmas.helpers.inout import add_records, dump, load
from negmas.helpers.strings import humanize_time
from negmas.helpers.types import get_class, get_full_type_name, instantiate
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators.negotiator import Negotiator
from negmas.preferences.ops import nash_point, pareto_frontier
from negmas.sao import SAOMechanism
import pandas as pd
from pandas.errors import EmptyDataError
from rich import print
from rich.progress import Progress, track
import typer

from utils import clean_results, full_name, get_rationality, nash_dist, pareto_dist


NTRIALS = 3
NDOMAINS = None
SAO_TIME_LIMIT = 3 * 60
SAO_ROUNDS = 2.0
# NTRIALS = 1
# NDOMAINS = 2
# SAO_TIME_LIMIT = 10
MECHANISMS: dict[str, Type[Mechanism]] = dict(AO=SAOMechanism, TAU=TAUMechanism)
RESULTS_FILE_NAME = f"results.csv"


def safediv(a, b):
    return (a / b) if abs(b) > 1e-13 else 1.0 if abs(a) < 1e-13 else float("inf")


@dataclass
class RunInfo:
    mechanism_name: str
    strategy_name: str
    domain_name: str
    mechanism: str
    strategy: str
    domain: Scenario
    domain_info: dict[str, Any]
    trial: int
    time_limit: int
    n_rounds: int
    mechanism_params: dict[str, Any]
    reserved0: float
    reserved1: float


def read_found(
    file_name, remove_failed: bool = True
) -> tuple[dict[tuple[str, str, str, str, str], int], list[dict]]:
    path = Path(file_name)
    if not path.exists():
        return defaultdict(int), []
    try:
        data: pd.DataFrame = pd.read_csv(path)  # type: ignore
    except EmptyDataError:
        return defaultdict(int), []
    records = data[~data["failed"]].to_dict(orient="records")
    found = defaultdict(int)
    for record in records:
        if remove_failed and "failed" in record and record["failed"]:
            continue
        found[
            (
                record["mechanism_name"],
                record["strategy_name"],
                record["domain_name"],
                f'{record.get("reserved0", 0):3.2f}',
                f'{record.get("reserved1", 0):3.2f}',
            )
        ] += 1
    return found, records


def run_once(info: RunInfo, file_name: str) -> dict[str, Any]:
    print(
        f"Running: {info.mechanism_name} {info.strategy_name} on {info.domain_name} ({info.reserved0}, {info.reserved1}) ... start at {datetime.now()}",
        flush=True,
    )
    pareto = info.domain_info["pareto_utils"]
    info.domain.ufuns[0].reserved_value = info.reserved0
    info.domain.ufuns[1].reserved_value = info.reserved1
    maxwelfare = sum(pareto[0])
    nashwelfare = info.domain_info["nash_welfare"]
    _full_start = perf_counter()
    m = instantiate(
        info.mechanism, outcome_space=info.domain.agenda, **info.mechanism_params
    )
    agreement = None
    try:
        failed = False
        _start = perf_counter()
        for ufun in info.domain.ufuns:
            m.add(instantiate(info.strategy), ufun=ufun)
        _start_run = perf_counter()
        m.run()
        elapsed = perf_counter() - _start
        elapsed_run = perf_counter() - _start_run
        agreement = m.agreement
        agreement_utils = tuple(ufun(agreement) for ufun in info.domain.ufuns)
        elapsed_full = perf_counter() - _full_start
        welfare = sum(agreement_utils)
        result = dict(
            mechanism_name=info.mechanism_name,
            strategy_name=info.strategy_name,
            domain_name=info.domain_info["name"],
            n_outcomes=info.domain_info["n_outcomes"],
            opposition=info.domain_info["opposition"],
            max_welfare=maxwelfare,
            time=elapsed,
            full_time=elapsed_full,
            run_time=elapsed_run,
            succeeded=agreement is not None,
            agreement_utils=agreement_utils,
            welfare=welfare,
            relative_welfare=safediv(welfare, maxwelfare),
            pdist=pareto_dist(agreement_utils, pareto),
            ndist=nash_dist(agreement_utils, info.domain_info["nash_utils"]),
            time_limit=info.time_limit,
            n_rounds=info.n_rounds,
            nash_welfare=nashwelfare,
            relative_nash_welfare=safediv(welfare, nashwelfare),
            reserved0=info.domain.ufuns[0].reserved_value,
            reserved1=info.domain.ufuns[1].reserved_value,
            reserved=tuple(u.reserved_value for u in info.domain.ufuns),
            steps=m.current_step,
            failed=False,
        )
    except Exception as e:
        failed = True
        print(
            f"[red]Failed: {info.mechanism_name}, {info.strategy_name}, {info.domain_name}[/red]: {str(e)}"
        )
        elapsed_full = perf_counter() - _full_start
        agreement_utils = tuple(0.0 for _ in info.domain.ufuns)
        welfare = sum(agreement_utils)
        result = dict(
            mechanism_name=info.mechanism_name,
            strategy_name=info.strategy_name,
            domain_name=info.domain_info["name"],
            n_outcomes=info.domain_info["n_outcomes"],
            opposition=info.domain_info["opposition"],
            max_welfare=maxwelfare,
            time=None,
            full_time=elapsed_full,
            run_time=None,
            succeeded=None,
            agreement_utils=agreement_utils,
            welfare=welfare,
            relative_welfare=safediv(welfare, maxwelfare),
            pdist=pareto_dist(agreement_utils, pareto),
            ndist=nash_dist(agreement_utils, info.domain_info["nash_utils"]),
            time_limit=info.time_limit,
            n_rounds=info.n_rounds,
            nash_welfare=nashwelfare,
            relative_nash_welfare=safediv(welfare, nashwelfare),
            reserved0=info.domain.ufuns[0].reserved_value,
            reserved1=info.domain.ufuns[1].reserved_value,
            reserved=tuple(u.reserved_value for u in info.domain.ufuns),
            steps=m.current_step,
            failed=True,
        )
    for k in ("n_outcomes", "opposition", "nash_welfare"):
        result[k] = info.domain_info[k]

    add_records(file_name, [result])
    cond = (
        "[red]Fail[/red]"
        if failed
        else "[yellow]None[/yellow]"
        if not agreement
        else "[green]OK[/green]"
    )

    print(
        f"{cond}: {info.mechanism_name} {info.strategy_name} on {info.domain_name} ({info.reserved0}, {info.reserved1}) ... DONE at {datetime.now()} (in {humanize_time(elapsed_full)})",
        flush=True,
    )
    return result


def add_pareto(dinfo: dict[str, Any], d: Scenario) -> dict[str, Any]:
    outcomes = list(d.agenda.enumerate_or_sample())
    utils, indices = pareto_frontier(d.ufuns, outcomes, sort_by_welfare=True)
    dinfo["pareto_utils"] = utils
    dinfo["pareto_indices"] = indices
    dinfo["pareto_outcomes"] = [outcomes[_] for _ in indices]
    np, ni = nash_point(d.ufuns, utils, d.agenda)
    if np is not None and ni is not None:
        dinfo["nash_utils"] = np
        dinfo["nash_welfare"] = sum(np)
        dinfo["nash_index"] = ni
        dinfo["nash_outcome"] = outcomes[ni]
    return dinfo


def get_reserved(domain: Scenario, r0: float, r1: float) -> tuple[float, float]:
    outcomes = list(domain.agenda.enumerate_or_sample())
    results, r = [], [r0, r1]
    for i in range(2):
        u = domain.ufuns[i]
        utils = sorted([u(_) for _ in outcomes], reverse=True)
        limit = int(r[0] * len(utils))
        results.append(utils[limit] - 0.0001)
    return tuple(results)


def run_all(
    domains: dict[str, Scenario],
    n_trials: int,
    time_limit: int,
    n_rounds: float,
    reserved0: list[float],
    reserved1: list[float],
    file_name: str,
    serial: bool,
    strategies: dict[str, tuple[Type[Negotiator], ...]],
    tauinf: bool,
    tau0: bool,
    ignore_irrational: bool,
    max_cores: int,
    order=False,
    reversed=False,
    rationality: dict[tuple[str, int, int], bool] = dict(),
    mechanisms: dict[str, Type[Mechanism], ...] = MECHANISMS,  # type: ignore
):
    runs = []
    found, results = read_found(file_name)
    n_total = 0
    n_estimated_total = (
        (int(time_limit > 0) + int(n_rounds > 0))
        * len(strategies["AO"])
        * len(domains)
        * n_trials
        * max(1, len(reserved0) * len(reserved1))
    )
    n_estimated_total += (
        (int(tauinf) + int(tau0))
        * len(strategies["TAU"])
        * len(domains)
        * n_trials
        * max(1, len(reserved0) * len(reserved1))
    )
    print(f"Estimated {n_estimated_total} total runs")

    for dn, d in track(
        domains.items(), total=len(domains), description="Updating Stats"
    ):
        stats_path = Path(dn) / "stats.json"
        dinfo = load(stats_path)
        if "pareto" not in dinfo.keys():
            dinfo = add_pareto(dinfo, d)
            dump(dinfo, stats_path)
        r0d, r1d = d.ufuns[0].reserved_value, d.ufuns[1].reserved_value
        reserved_vals = list(product(reserved0, reserved1))
        if not reserved_vals:
            reserved_vals.append((r0d, r1d))
        n_steps = int(n_rounds * dinfo["n_outcomes"])

        for mn, m in mechanisms.items():
            if issubclass(m, SAOMechanism):
                params = []
                if time_limit > 0:
                    params.append(
                        dict(
                            time_limit=time_limit,
                            n_steps=None,
                            name="AOt",
                            extra_callbacks=True,
                        )
                    )
                if n_steps > 0:
                    params.append(
                        dict(
                            n_steps=n_steps,
                            hidden_time_limit=6 * 60,
                            time_limit=None,
                            name="AOr",
                            extra_callbacks=True,
                        )
                    )
            else:
                params = []
                if tau0:
                    params.append(dict(min_unique=0, name="TAU0"))
                if tauinf:
                    params.append(dict(min_unique=float("inf"), name="TAUinf"))
            for p in params:
                new_mn: str = p.pop("name")  # type: ignore
                for s in strategies[mn]:
                    sn = s.__name__
                    for r0, r1 in reserved_vals:
                        if ignore_irrational and not rationality.get(
                            (dn.split("/")[-1], int(r0 * 10 + 0.5), int(r1 * 10 + 0.5)),
                            True,
                        ):
                            print(
                                f"    [cyan]Ignoring[/cyan] {(dn.split('/')[-1], int(r0 * 10 + 0.5), int(r1*10 + 0.5))}",
                                flush=True,
                            )
                            continue
                        n_remaining = (
                            n_trials
                            - found[
                                (
                                    new_mn,
                                    sn,
                                    dn.split("/")[-1],
                                    f"{r0:3.2f}",
                                    f"{r1:3.2f}",
                                )
                            ]
                        )
                        n_total += n_trials
                        if n_remaining < 1:
                            continue
                        for t in range(n_remaining):
                            runs.append(
                                RunInfo(
                                    mechanism_name=new_mn,
                                    strategy_name=sn,
                                    domain_name=dn,
                                    mechanism=get_full_type_name(m),
                                    strategy=get_full_type_name(s),
                                    domain=d,
                                    domain_info=dinfo,
                                    trial=t,
                                    time_limit=time_limit,
                                    n_rounds=n_steps,
                                    mechanism_params=p,
                                    reserved0=r0,
                                    reserved1=r1,
                                )
                            )
    n = len(runs)
    if n < 0:
        print("Nothing to run")
        return
    if order:
        runs = sorted(
            runs, key=lambda x: (x.mechanism_name, x.strategy_name, x.domain_name)
        )
        if reversed:
            runs.reverse()
    else:
        shuffle(runs)
    print(f"Submitting {len(runs)} tasks (of {n_total})", flush=True)
    if serial:
        for info in track(runs, total=len(runs)):
            results.append(run_once(info, file_name))
    else:
        futures = []
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with Progress() as progress:
            submissions = progress.add_task("Making Negotiations ...", total=n)
            with ProcessPoolExecutor(max_workers=cpus) as pool:
                for i, info in enumerate(runs):
                    futures.append(pool.submit(run_once, info, file_name))
                    progress.update(submissions, completed=(i + 1) / n)
                    progress.refresh()
        with Progress() as progress:
            print("Running ...", flush=True)
            negotiations = progress.add_task("Executing Negotiations ...", total=n)
            for i, f in enumerate(as_completed(futures)):
                results.append(f.result())
                progress.update(negotiations, completed=(i + 1) / n)
                progress.refresh()

    # print("Saving ...")
    # pd.DataFrame.from_records(results).to_csv(file_name, index=False)
    print("DONE")


def main(
    trials: int = NTRIALS,
    domains: int = sys.maxsize,
    timelimit: int = SAO_TIME_LIMIT,
    rounds: float = SAO_ROUNDS,
    restart: bool = False,
    small: bool = False,
    serial: bool = False,
    reserved0: list[float] = [],
    reserved1: list[float] = [],
    file_name: str = RESULTS_FILE_NAME,
    ao: list[str] = [
        "Atlas3",
        "AgentK",
        "Caduceus",
        "HardHeaded",
        "CUHKAgent",
        "NiceTitForTat",
        "AspirationNegotiator",
        "MiCRONegotiator",
    ],
    tau0: bool = True,
    tauinf: bool = True,
    order: bool = False,
    reversed: bool = True,
    ignore_irrational: bool = True,
    max_cores: int = 0,
    add_host_name: bool = False,
    domains_path: Path = Path("domains"),
):
    if small:
        domains = min(domains, 3)
        trials = 1
        timelimit = 10
        rounds = 10
    if add_host_name:
        file_name = f"{Path(file_name).stem}-{gethostname()}.csv"
    else:
        file_name = f"{Path(file_name).stem}.csv"

    print(f"Will save results to: {file_name}")
    clean_results(Path(file_name))
    if restart and Path(file_name).exists():
        print("Removing old runs")
        unlink(file_name)

    files = tuple(sorted(domains_path.glob("*")))
    if domains is not None and domains < len(files):
        files = files[:domains]

    all_domains = {
        name: s
        for name, s in (
            (
                str(f),
                Scenario.from_genius_folder(
                    f, ignore_discount=True, ignore_reserved=False
                ),
            )
            for f in track(files, total=len(files), description="Loading Domains")
            if f.is_dir()
        )
        if s is not None
    }
    if "none" in ao:
        aostrategies = []
    else:
        aostrategies = tuple(get_class(full_name(_)) for _ in ao)
    strategies: dict[str, tuple[Type[Negotiator], ...]] = dict(  # type: ignore
        AO=aostrategies,
        TAU=(SCSNegotiator,),
    )
    run_all(
        domains=all_domains,
        n_trials=trials,
        time_limit=timelimit,
        n_rounds=rounds,
        serial=serial,
        reserved0=reserved0,
        reserved1=reserved1,
        file_name=file_name,
        mechanisms=MECHANISMS,
        strategies=strategies,
        tau0=tau0,
        tauinf=tauinf,
        ignore_irrational=ignore_irrational,
        max_cores=max_cores,
        order=order,
        reversed=reversed,
        rationality=get_rationality(),
    )


if __name__ == "__main__":

    typer.run(main)
