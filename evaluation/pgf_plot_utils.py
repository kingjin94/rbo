from typing import Collection

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def bootstrap_line_plot(
        data: pd.DataFrame,
        x_axis: str,
        fig_column: str,
        fig_hue: str,
        y_axis: str,
        bootstrap_over: Collection[str],
        statistic=np.mean,
        n_resamples: int = 1000,
        confidence_level: float = 0.95,
        rng=np.random.default_rng()):
    """
    Create a dataframe that pgf plot can use to create a line plot with confidence intervals.

    The created dataframe will have the following columns:
    - fig_column.fig_hue.x
    - fig_column.fig_hue.mean
    - fig_column.fig_hue.low
    - fig_column.fig_hue.high

    :param data: The data to plot, assumed with columns
    :param x_axis: The column to use as the x axis, e.g., time step.
    :param fig_column: The column to use for the figure, e.g., task set.
    :param fig_hue: The column to use for the hue, e.g., optimization scope.
    :param y_axis: The column to use as the y axis, e.g., fitness value.
    :param bootstrap_over: The columns to bootstrap over, e.g., seeds or specific tasks.
    :param statistic: The statistic to calculate, default: mean.
    :param n_resamples: The number of resamples to perform for each time step.
    :param confidence_level: The confidence level to use for the confidence interval.
    """
    if x_axis not in data.columns:
        raise ValueError(f"Column {x_axis} not found in data.")
    for group in bootstrap_over:
        if group not in data.columns:
            raise ValueError(f"Column {group} not found in data.")
    if fig_column not in data.columns:
        raise ValueError(f"Column {fig_column} not found in data.")
    if fig_hue not in data.columns:
        raise ValueError(f"Column {fig_hue} not found in data.")
    if y_axis not in data.columns:
        raise ValueError(f"Column {y_axis} not found in data.")

    ret_columns = {}
    for col_name, col_group in data.groupby(fig_column):
        for hue_name, hue_group in col_group.groupby(fig_hue):
            line_prefix = f"{col_name}.{hue_name}."
            y_df = hue_group.pivot_table(index=x_axis, columns=bootstrap_over, values=y_axis)
            y_values = y_df.to_numpy(dtype=np.float64)
            y_values += 1e-9 * rng.random(y_values.shape)  # Add small noise to avoid degenerate data with 0 variance
            bs_res = bootstrap(y_values[np.newaxis, :], statistic, n_resamples=n_resamples,
                               confidence_level=confidence_level, axis=-1)
            ret_columns[line_prefix + 'x'] = y_df.index.map(pd.Timestamp).map(pd.Timestamp.timestamp).to_numpy()
            ret_columns[line_prefix + 'mean'] = y_df.mean(axis='columns').to_numpy(dtype=np.float64)
            ret_columns[line_prefix + 'low'] = bs_res.confidence_interval.low
            ret_columns[line_prefix + 'high'] = bs_res.confidence_interval.high

    return pd.DataFrame(ret_columns)
