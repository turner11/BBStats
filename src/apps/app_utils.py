from __future__ import annotations

import itertools
import re
from collections import defaultdict
from io import StringIO
from pathlib import Path
import numpy as np

import pandas as pd
import gsheetsdb
import cachetools.func
from datetime import datetime, timedelta
from datetime import time
import logging

logger = logging.getLogger(__name__)

now = datetime.now()
# Create a connection object.
conn = gsheetsdb.connect()

DEFAULT_MINUTES_IN_QUARTER = 10.0

renames = {'time_left': 'time',
           'points': 'team',
           'points_against': 'opponent'}


@cachetools.func.ttl_cache(maxsize=10, ttl=15)
def get_sheets_data(sheets_url, headers=1) -> pd.DataFrame:
    query = f'SELECT * FROM "{sheets_url}"'
    cursor = conn.execute(query, headers=headers)
    df = pd.DataFrame(cursor.fetchall())
    return df.copy()


def _resolve_path_arg(data_arg):
    path_arg = data_arg
    if isinstance(data_arg, str):
        if data_arg.lower().startswith('http'):
            # noinspection PyCallingNonCallable
            path_arg = get_sheets_data(data_arg)
        elif Path(data_arg).exists():
            path_arg = data_arg
        else:
            # noinspection PyTypeChecker
            df = pd.read_csv(StringIO(path_arg.strip()), )
            path_arg = df

    return path_arg


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


def get_stats_from_raw_data(df, group_size):
    combinations_by_snapshot = df.players.apply(lambda lu: tuple(itertools.combinations(lu, group_size))).values
    played_groups = set(itertools.chain.from_iterable(combinations_by_snapshot))

    dfs = []
    for line_up in played_groups:
        line_up = set(line_up)
        indices = df.players.apply(lambda ps: set(ps).intersection(line_up) == line_up)
        sum_cols = [c for c in df.columns if c.endswith('diff')] + ['elapsed']

        df_group = df[indices].agg({c: sum for c in sum_cols})
        try:
            # series
            df_group = df_group.to_frame().T
        except AttributeError:
            # data frame
            pass
        df_group = df_group.assign(players=[sorted(line_up)])
        dfs.append(df_group)

    df_stats = pd.concat(dfs).reset_index(drop=True)

    elapsed_minutes = df_stats.elapsed
    elapsed_seconds = elapsed_minutes * 60
    df_stats['played'] = get_friendly_time(elapsed_seconds)
    df_stats['score_pm'] = df_stats.score_diff / elapsed_minutes
    df_stats['offense_pm'] = df_stats.offense_diff / elapsed_minutes
    df_stats['defence_pm'] = df_stats.defence_diff / elapsed_minutes

    types = {c: t for c, t in df.dtypes.items() if c in df_stats.columns}
    df_stats = df_stats.astype(types)

    leading_cols = ['score_diff', 'score_pm']
    last_cols = ['players', 'elapsed']
    mid_clos = [c for c in df_stats.columns if c not in leading_cols and c not in last_cols]
    mid_clos = sorted(mid_clos, key=lambda c: c)
    cols = leading_cols + mid_clos + last_cols
    return df_stats[cols].sort_values('score_pm', ascending=False).copy()


# noinspection PyCallingNonCallable
# @cachetools.func.ttl_cache(maxsize=10, ttl=60*5)
def get_player_images(players_url) -> defaultdict:
    no_image = 'https://media.istockphoto.com/vectors/basketball-player-standing-and-holding-ball-vector-silhouette-vector-id1299295749?k=20&m=1299295749&s=170667a&w=0&h=D8t1TTfMp_E-W7Vn-HBqRqpfsbCb4QxuR5lthFBq0fs='
    images = defaultdict(lambda: no_image)
    try:
        df = get_sheets_data(players_url)
        df['image'] = df.image.fillna(no_image)
        players_images = df.set_index('player').image.to_dict()
        images.update(players_images)
    except Exception:
        logger.exception('Failed to get player images')
    return images
