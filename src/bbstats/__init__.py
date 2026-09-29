from __future__ import annotations

import itertools
import re
from io import StringIO
from pathlib import Path
import numpy as np

import pandas as pd
import urllib.request
from datetime import timedelta
from datetime import time


DEFAULT_MINUTES_IN_QUARTER = 10.0

renames = {'time_left': 'time',
           'points': 'team',
           'points_against': 'opponent'}


SORTS = {'top': ('score_pm', False), 'offense': ('offense_pm', False), 'defense': ('defence_pm', True)}


def to_csv_url(url: str) -> str:
    """Google Sheets share URL -> CSV export URL (needs 'anyone with the link'). Other URLs pass through."""
    m = re.match(r'(https://docs\.google\.com/spreadsheets/d/[^/?#]+)', url)
    if not m:
        return url
    gids = re.findall(r'gid=(\d+)', url)  # last one: the #gid fragment names the tab
    return f'{m.group(1)}/export?format=csv' + (f'&gid={gids[-1]}' if gids else '')


def fetch_csv(url: str, timeout=15) -> str:
    with urllib.request.urlopen(to_csv_url(url), timeout=timeout) as resp:
        return resp.read().decode('utf-8')


def _resolve_path_arg(data_arg):
    if not isinstance(data_arg, str):
        return data_arg
    s = data_arg.strip()
    if '\n' in s:
        return pd.read_csv(StringIO(s))
    if s.lower().startswith('http'):
        return pd.read_csv(StringIO(fetch_csv(s)))
    return s if Path(s).exists() else pd.read_csv(StringIO(s))


def get_snapshots_df(path_arg: str | Path | pd.DataFrame, minutes_in_quarter=DEFAULT_MINUTES_IN_QUARTER):
    minutes_in_quarter = minutes_in_quarter or DEFAULT_MINUTES_IN_QUARTER
    path_arg = _resolve_path_arg(path_arg)
    df_raw = _load_raw_data(path_arg, minutes_in_quarter=minutes_in_quarter)
    df = _enrich_data(df_raw, minutes_in_quarter=minutes_in_quarter)
    return df


def get_stats_df(data_arg, group_size, minutes_in_quarter=DEFAULT_MINUTES_IN_QUARTER):
    df = get_snapshots_df(data_arg, minutes_in_quarter=minutes_in_quarter)
    df_stats = get_stats_from_raw_data(df, group_size)
    return df_stats


def _enrich_data(df_raw, minutes_in_quarter=DEFAULT_MINUTES_IN_QUARTER):
    if df_raw is None:
        raise ValueError('Cannot enrich None dataframe')
    if not len(df_raw):
        return df_raw

    df = df_raw.copy()
    # Game Time
    time_left_in_quarter = df.time
    minutes_left_in_quarter = time_left_in_quarter.dt.total_seconds() / 60.0
    max_minutes = minutes_left_in_quarter.max()
    if max_minutes > minutes_in_quarter:
        raise ValueError(f'Got records with more time ({max_minutes}) than minutes in quarter {minutes_in_quarter}')
    minutes_in_future_quarters = (4 - df.quarter) * minutes_in_quarter
    minutes_left_in_game = minutes_in_future_quarters + minutes_left_in_quarter

    df['game_time_left'] = minutes_left_in_game

    # Elapsed
    elapsed = df.game_time_left - df.game_time_left.shift(-1)
    df['elapsed'] = elapsed.fillna(0)

    # Score diff
    offense_diff = df.team.shift(-1) - df.team
    defence_diff = df.opponent.shift(-1) - df.opponent
    df['offense_diff'] = offense_diff.fillna(0).astype(int)
    df['defence_diff'] = defence_diff.fillna(0).astype(int)

    df['score_diff'] = df.offense_diff - df.defence_diff

    df['players'] = df.apply(lambda row: [row.player_1, row.player_2, row.player_3, row.player_4, row.player_5], axis=1)

    return df


def _load_raw_data(path_arg: str | Path | pd.DataFrame, minutes_in_quarter=DEFAULT_MINUTES_IN_QUARTER):
    if isinstance(path_arg, pd.DataFrame):
        df = path_arg.copy()
    elif isinstance(path_arg, str):
        df = pd.read_excel(str(path_arg))
    else:
        raise TypeError(f'Cannot parse data of type {type(path_arg).__name__}')

    df = df[[c for c in df.columns if 'unnamed' not in c.lower()]]
    space_renames = {c: c.replace(' ', '_').replace('#', 'player_').lower() for c in df.columns}
    df: pd.DataFrame = df.rename(columns=space_renames)

    df.rename(columns={d: d.strip() for d in df.columns})
    df.infer_objects()
    df = df.rename(columns=renames)
    df = df.dropna(subset=['time'], how='all')
    df['time'] = df.time.apply(get_time)
    df['auto_added'] = False

    # Add record for each quarter start
    dfs_q = []
    for quarter, dfq in df.groupby('quarter'):
        min_time_idx = dfq.time.idxmax()
        first_record = dfq.loc[min_time_idx]
        first_record_time = first_record.time.to_pytimedelta().total_seconds()
        if first_record_time != minutes_in_quarter * 60:
            new_first = pd.Series(first_record)
            new_first['time'] = timedelta(seconds=minutes_in_quarter * 60)
            new_first['team'] = np.nan
            new_first['opponent'] = np.nan
            new_first['auto_added'] = True
            dfq = pd.concat([dfq, new_first.to_frame().T], ignore_index=True)
        dfs_q.append(dfq)
    df = pd.concat(dfs_q)

    df = df.sort_values(['quarter', 'time'], ascending=(True, False)).reset_index(drop=True)
    df = df.ffill()
    df.loc[0, ['team']] = df.loc[0, ['team']].fillna(0)
    df.loc[0, ['opponent']] = df.loc[0, ['opponent']].fillna(0)

    df['time'] = pd.to_timedelta(df.time)
    df['auto_added'] = df.auto_added.astype(bool)
    df['friendly_time'] = get_friendly_time(df.time)
    df['quarter'] = df.quarter.astype(int)

    df['team'] = df.team.astype(int)
    df['opponent'] = df.opponent.astype(int)

    for col in df.columns:
        if col.startswith('player_'):
            df[col] = df[col].ffill().fillna(-1).astype(int)
    return df.reset_index(drop=True).copy()


def get_friendly_time(time_series: pd.Series) -> pd.Series:
    """
    Gets the string representation of time
    :param time_series: a series of  time delta / floats that represents seconds
    :return: a series with the string representation of time input
    """
    try:
        total_seconds = time_series.dt.total_seconds()
    except AttributeError:
        total_seconds = time_series

    minutes = (total_seconds / 60).astype(int)
    seconds = (total_seconds - minutes * 60).astype(int)
    str_minutes = minutes.astype(str)
    str_seconds = seconds.astype(str)
    str_minutes, str_seconds = [s.str.pad(2, side='left', fillchar='0') for s in (str_minutes, str_seconds)]
    friendly_time = str_minutes + ':' + str_seconds
    return friendly_time


def get_time(raw_hour):
    # noinspection PyBroadException
    try:
        # if isinstance(raw_hour, datetime):
        if isinstance(raw_hour, time):
            time_stamp = raw_hour
        elif isinstance(raw_hour, str):
            clean_time = re.sub("[^0-9:]", "", raw_hour)
            args = [int(v) for v in clean_time.split(':')[-2:]]
            time_stamp = time(minute=args[0], second=args[1])
        elif isinstance(raw_hour, pd.Timestamp):
            # The expected format for time remaining is minutes:seconds , but excel/ sheets parses as hours:minutes
            time_stamp = time(minute=raw_hour.hour, second=raw_hour.minute)
        elif isinstance(raw_hour, float) and np.isnan(raw_hour):
            time_stamp = None
        else:
            raise NotImplementedError('Check delta from what is this...')

        if time_stamp is not None:
            out_date = timedelta(minutes=time_stamp.minute, seconds=time_stamp.second)
        else:
            out_date = None

    except Exception:
        out_date = None
    return out_date


def get_stats_from_raw_data(df, group_size, sort='top'):
    """Lineup stats over snapshots. Needs columns: players, elapsed, offense_diff, defence_diff."""
    if sort not in SORTS:
        raise ValueError(f'Unknown sort {sort!r}, expected one of {list(SORTS)}')
    col, ascending = SORTS[sort]
    sum_cols = ['offense_diff', 'defence_diff', 'elapsed']
    combinations_by_snapshot = df.players.apply(lambda lu: tuple(itertools.combinations(lu, group_size))).values
    played_groups = set(itertools.chain.from_iterable(combinations_by_snapshot))

    rows = []
    for line_up in played_groups:
        line_up = set(line_up)
        indices = df.players.apply(lambda ps: set(ps).intersection(line_up) == line_up)
        totals = df[indices][sum_cols].sum()
        if totals.elapsed > 0:
            rows.append({**totals.to_dict(), 'players': sorted(line_up)})

    df_stats = pd.DataFrame(rows, columns=sum_cols + ['players'])
    df_stats['score_diff'] = df_stats.offense_diff - df_stats.defence_diff
    df_stats['played'] = get_friendly_time(df_stats.elapsed * 60)
    for name, diff in (('score_pm', 'score_diff'), ('offense_pm', 'offense_diff'), ('defence_pm', 'defence_diff')):
        df_stats[name] = df_stats[diff] / df_stats.elapsed

    leading_cols = ['score_diff', 'score_pm']
    last_cols = ['players', 'elapsed']
    mid_cols = sorted(c for c in df_stats.columns if c not in leading_cols + last_cols)
    return df_stats[leading_cols + mid_cols + last_cols].sort_values(col, ascending=ascending).reset_index(drop=True)
