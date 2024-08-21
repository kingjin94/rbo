from typing import List

import pandas as pd


FAIL_REASON_ORDER = [
    'Robot Length Filter',
    'Simple IK Filter',
    'IK Filter',
    'Path planning failed',
    'Path planning timeout',
    'Invalid Path',
    'No failure']


def run_eval():
    """Print out a suggestion to run the evaluation script with GNU parallel."""
    print("Please look into experiments/run_evaluation.sh")


def cleanup_preprocess_results(result_df: pd.DataFrame, group_by: List[str] = None) -> pd.DataFrame:
    """
    Clean up and preprocess the results.

    * Turn 'Run Time' into pandas time
    * Add 'Maximum Reward' column which is the maximum reward up to the current step
    * Rename 'Unnamed: 0' to 'Step'
    * Add 'Step Time' column which is the time taken from the previous step
    * Add 'Success Till Step' column which is True if a valid solution was found up to the current step
    * Rename 'Algorithm' to shorter names
    * Clean up failure reasons
    """
    if group_by is None:
        group_by = ['Algorithm', 'Task ID', 'Seed']

    # Turn run-time into time
    result_df['Run Time'] = pd.to_datetime(result_df['Run Time'], unit="s")
    # Maximum reward up to step
    result_df['Maximum Reward'] = result_df.groupby(group_by)['Reward'].cummax()
    # Rename step
    result_df = result_df.rename(columns={'Unnamed: 0': "Step"})
    # Time per step
    result_df['Step Time'] = result_df.groupby(group_by)['Run Time'].diff()
    # Fill in first step assuming start at 0
    result_df.loc[result_df['Step'] == 0, 'Step Time'] = result_df[result_df['Step'] == 0]['Run Time'] - pd.Timestamp(0)
    result_df['Step Time'] += pd.Timestamp(0)  # Make plotable
    # Success up to step
    result_df['Success Till Step'] = result_df.groupby(group_by)['Valid Solution'].cummax()
    # Rename algorithm
    result_df['Algorithm'] = result_df['Algorithm'].replace({'BOOptimizer': 'BO',
                                                             'GAOptimizer': 'GA',
                                                             'RandomBaseOptimizer': 'Random',
                                                             'AdamOptimizer': 'SGD',
                                                             'DummyOptimizer': 'Dummy'})
    # Clean up failure reasons
    result_df['Fail Reason'] = result_df['Fail Reason'].str.replace(
        r' \(<timor.task.Task.Task object at 0x[0-9a-f]*>\)', '', regex=True)  # Remove task object address
    result_df['Fail Reason'] = result_df['Fail Reason'].str.replace(
        r"'task': Task [A-Za-z0-9_/]+,", "'task': ..,", regex=True)  # Remove task id
    result_df['Fail Reason'] = result_df['Fail Reason'].str.replace(
        r"Unknown; info: \{'is_success': False.*\}", "Invalid path", regex=True)  # Remove solution for failed tasks
    result_df['Fail Reason'] = result_df['Fail Reason'].replace({
        "Failed filters: [InverseKinematicsSolvable({'ignore_self_collisions': True, 'max_iter': 300})]":
            'Simple IK Filter',
        "Failed filters: [RobotLongEnoughFilter]": "Robot Length Filter",
        "Trajectory generation failed": "Path planning failed",
        "Failed filters: [InverseKinematicsSolvable({'task': .., 'max_iter': 1500})]": "IK Filter",
        "": "No failure",
        "Path planning failed": "Path planning failed",
        "Timeout task solver": "Path planning timeout",
    })
    return result_df


def normalize_time(result_df: pd.DataFrame, sampling_time: str = '1000ms',
                   group_by: List[str] = None) -> pd.DataFrame:
    """Return resampled DataFrame with common frequency and padded end time."""
    if group_by is None:
        group_by = ['Algorithm', 'Task ID', 'Seed']
    max_run_time = result_df['Run Time'].max()
    final_dfs = []
    for group_idx, group in result_df.groupby(group_by):
        # Pad start
        start_df = group[group['Step'] == 0].copy()
        start_df['Run Time'] = pd.Timestamp(0)
        start_df['Reward'] = start_df['Reward Fail']
        start_df['Solution'] = None
        start_df['Valid Solution'] = False
        start_df['Fail Reason'] = ''
        start_df['Success Till Step'] = False
        # Pad end
        end_df = group[group['Step'] == group['Step'].max()].copy()
        end_df['Run Time'] = max_run_time + pd.Timedelta(1, unit="ms")
        padded_df = pd.concat([start_df, group, end_df])
        final_dfs.append(padded_df.set_index('Run Time').resample(sampling_time).ffill().reset_index())
    return pd.concat(final_dfs, axis=0, ignore_index=True)


def print_step_count(result_df):
    """Print step count statistics per algorithm."""
    final_step_count_per_trial = result_df.groupby(['Algorithm', 'Task ID', 'Seed'])['Step'].max()
    step_count_stats = final_step_count_per_trial.groupby(['Algorithm']).aggregate(['mean', 'max', 'min'])
    print("Step count statistics:")
    print(step_count_stats)
