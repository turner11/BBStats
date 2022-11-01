from __future__ import annotations

import dataclasses
from collections import defaultdict
import pandas as pd
import streamlit as st
import logging
import coloredlogs
import sys
from urllib.parse import urldefrag
import app_utils

logger = logging.getLogger(__name__)

PARAM_NAME_URL = 'data'
PARAM_PLAYERS_SHEET_ID = 'players'


@dataclasses.dataclass(frozen=True)
class Inputs:
    df: pd.DataFrame
    group_size: int
    all_players: list[int]
    players_images: defaultdict


@dataclasses.dataclass
class AppParams:
    url: str | pd.DataFrame
    players_url: str
    minutes_in_quarter: int = 10


def get_df(url='', minutes_in_quarter=None):
    path_arg = st.sidebar.text_input('URL', url)
    df = app_utils.get_snapshots_df(path_arg, minutes_in_quarter=minutes_in_quarter)
    return df, path_arg


def get_inputs(params: AppParams = None):
    with st.expander('Settings'):
        quarter_options = [params.minutes_in_quarter] + [m for m in (10, 12) if m != params.minutes_in_quarter]
        minutes_in_quarter = st.radio('Minutes in quarter', quarter_options, )
        params.minutes_in_quarter = minutes_in_quarter
    with st.spinner('getting data'):
        df, path = get_df(params.url, params.minutes_in_quarter)

    players_images = defaultdict
    if params.players_url:
        with st.spinner('getting players'):
            players_images = app_utils.get_player_images(params.players_url)

    st.markdown(f'[data source]({path})')

    all_players = df.players.apply(set)
    all_players = sorted(set.union(*all_players))

    group_count = st.number_input('group size', 1, 5, 5)
    # selected_players = st.multiselect('players', options=all_players, default=all_players)

    query_params = st.experimental_get_query_params()
    query_params.update({PARAM_NAME_URL: path})
    st.experimental_set_query_params(**query_params)
    return Inputs(df, group_count, all_players, players_images)


def display_lineups(df_stats, players_images):
    for row_idx, row in df_stats.iterrows():
        txt = f'{row.played} minutes, {row.offense_diff}:{row.defence_diff}'
        st.subheader(txt)
        st.dataframe(row.to_frame().T)
        container = st.container()
        columns = container.columns(len(row.players)) if len(row.players) > 1 else [container]
        for i, player in enumerate(row.players):
            col = columns[i]
            url = players_images[player]
            col.image(url, width=100)
            # col.markdown(f'![{player}]({url})')
            col.text(player)
        st.markdown('___')


def app(default_data_path=None):
    with st.sidebar:
        inputs = get_inputs(default_data_path)

    df = inputs.df
    with st.expander('raw data', expanded=True):
        df_display = df[[c for c in df.columns if c != 'time']]
        # df_display = df.astype(str)
        st.dataframe(df_display)

    with st.spinner('Calculating stats'), st.expander('stats', expanded=True):
        df_stats = app_utils.get_stats_from_raw_data(inputs.df, inputs.group_size)
        st.dataframe(df_stats)

    with st.expander('lineups', expanded=True):
        display_lineups(df_stats, inputs.players_images)


def main(use_external_parameters=False):
    logger_format = '%(asctime)s [%(threadName)s] %(module)s::%(funcName)s %(levelname)s - %(message)s'
    coloredlogs.install(level='CRITICAL')
    coloredlogs.install(level='DEBUG', fmt=logger_format, logger=logger, stream=sys.stdout, isatty=True)

    st.set_page_config(layout='wide', page_icon='🏀')

    if use_external_parameters:

        query_params = st.experimental_get_query_params() or defaultdict(list)

        params_url = query_params.get(PARAM_NAME_URL, [''])[0]
        if params_url:
            url = params_url
            players_sheet_id = query_params.get(PARAM_PLAYERS_SHEET_ID, [''])[0]
        else:
            url = st.secrets.get('data_path')
            players_sheet_id = st.secrets.get('players_sheet_id')

        st.experimental_set_query_params(**{PARAM_NAME_URL: url, PARAM_PLAYERS_SHEET_ID: players_sheet_id})

        url = url or ''
        players_url = ''

        if url and players_sheet_id:
            un_fragmented = urldefrag(url)
            players_url = f'{un_fragmented[0]}#gid={players_sheet_id}'

        params = AppParams(url=url, players_url=players_url)
    else:
        params = None

    import warnings
    with warnings.catch_warnings():
        # Suppressing pycharm warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        app(params)


if __name__ == '__main__':
    main(use_external_parameters=True)
