from pathlib import Path

import pandas as pd
from rich import print
import typer

from utils import (
    MAX_TRIALS,
    NONSTATS,
    PRECISION,
    STATS,
    do_all_tests,
    make_latex_table,
    read_data,
)


def main(
    files: list[Path] = [],
    failures: bool = False,
    rounds: bool = True,
    timelimit: bool = True,
    count: bool = False,
    agreements_only: bool = False,
    output: str = "tbl1",
    max_trials: int = MAX_TRIALS,
    insignificant: bool = False,
    significant: bool = True,
    allstats: bool = False,
    perdomain: bool = False,
    precision: int = PRECISION,
):
    if precision > 0:
        pd.set_option("display.precision", precision)
        pd.options.display.float_format = f"{{:.{precision}f}}".format
    data = read_data(files, failures, rounds, timelimit, agreements_only, max_trials)
    columns = STATS + NONSTATS
    filtered = data[columns]
    file_name = f"tables/{output}{'-perdomain' if perdomain else ''}.tex"
    print(
        make_latex_table(
            filtered, file_name, count, STATS, perdomain, precision=precision
        )
    )
    if not perdomain:
        do_all_tests(
            data,
            insignificant,
            allstats,
            basename="exp1",
            significant=significant,
            precision=precision,
        )


if __name__ == "__main__":
    typer.run(main)
