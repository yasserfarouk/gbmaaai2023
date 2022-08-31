#!/usr/bin/env python
from pathlib import Path
from shutil import copy

from negmas.helpers import unique_name
import pandas as pd
from pandas.errors import EmptyDataError
from rich import print
import typer

from utils import clean_results


def remove_extra(
    path: Path, max_trials: int, remove_failed: bool = True, verbose: bool = True
) -> pd.DataFrame | None:
    badcols = set(["index"] + [f"level_{i}" for i in range(10)])
    if not path.exists():
        return None
    try:
        data: pd.DataFrame = pd.read_csv(path)  # type: ignore
    except EmptyDataError:
        return None
    n_records = len(data)
    if verbose:
        print(f"  Found {n_records} records ... ", flush=True, end="")
    allcols = [_ for _ in data.columns if _ not in badcols]
    data = data[allcols]
    succeeded = data[~data["failed"]]
    failed = data[data["failed"]]

    # record["mechanism_name"],
    # record["strategy_name"],
    # record["domain_name"],
    # f'{record.get("reserved0", 0):3.2f}',
    # f'{record.get("reserved1", 0):3.2f}',

    def limit_trials(x: pd.DataFrame, n: int) -> pd.DataFrame:
        cols = [
            "mechanism_name",
            "strategy_name",
            "domain_name",
            "reserved0",
            "reserved1",
        ]
        y = x.groupby(cols).head(n).reset_index()
        if "index" in y.columns:
            y = y[[_ for _ in y.columns if _ != "index"]]
        return y

    succeeded = limit_trials(succeeded, max_trials)
    if not remove_failed:
        failed = limit_trials(failed, 1)
        succeeded = pd.concat((succeeded, failed))
    if verbose:
        if n_records == len(succeeded):
            print(f" [green]all OK[/green]", flush=True)
        else:
            print(
                f"remove [red]{n_records - len(succeeded)}[/red] keeping {len(succeeded)}",
                flush=True,
            )
    return succeeded


def process(f: Path, max_trials: int, remove_failed: bool, verbose: bool):
    print(f"Cleaning {f}")
    clean_results(f, verbose)
    if max_trials > 0:
        clean = remove_extra(f, max_trials, remove_failed, verbose)
        if clean is not None:
            backup = f.parent / "backups"
            backup.mkdir(exist_ok=True, parents=True)
            copy(f, backup / unique_name(f.name, rand_digits=1, sep=""))
            clean.to_csv(f, index=False)


def main(
    file_name: Path,
    pattern: str = "e*.csv",
    max_trials: int = 10,
    remove_failed: bool = False,
    verbose: bool = True,
):
    if not file_name.exists():
        return
    if file_name.is_dir():
        for f in file_name.glob(pattern):
            process(f, max_trials, remove_failed, verbose)
    else:
        process(file_name, max_trials, remove_failed, verbose)


if __name__ == "__main__":
    typer.run(main)
