import dataclasses
from collections import defaultdict
import pandas as pd
import streamlit as st
import logging
import coloredlogs
import sys
from pathlib import Path
import app_utils

logger = logging.getLogger(__name__)

PARAM_NAME_URL = 'data'


@dataclasses.dataclass(frozen=True)
class Inputs:
    df: pd.DataFrame
    group_size: int
    all_players: list[int]


def get_df(default_data_path=None, minutes_in_quarter=None):
    query_params = st.experimental_get_query_params() or defaultdict(list)
    params_priority = query_params.get(PARAM_NAME_URL, []) + [default_data_path, '']
    external_input = params_priority[0]
    path_arg = ''
    if external_input:
        path_arg = external_input
    if Path(path_arg).exists():
        path_arg = Path(path_arg).resolve()

    path_arg = st.sidebar.text_input('Excel path / URL', str(path_arg))

    df = app_utils.get_snapshots_df(path_arg, minutes_in_quarter=minutes_in_quarter)
    return df, path_arg


def get_inputs(default_data_path=None):
    with st.expander('Settings'):
        minutes_in_quarter = st.radio('Minutes in quarter', (10, 12), )
    with st.spinner('getting data'):
        df, path = get_df(default_data_path, minutes_in_quarter)
    st.markdown(f'[data source]({path})')

    all_players = df.players.apply(set)
    all_players = sorted(set.union(*all_players))

    group_count = st.number_input('group size', 1, 5, 5)
    # selected_players = st.multiselect('players', options=all_players, default=all_players)
    # st.experimental_set_query_params(**{PARAM_NAME_URL: path})
    return Inputs(df, group_count, all_players)


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


def main():
    logger_format = '%(asctime)s [%(threadName)s] %(module)s::%(funcName)s %(levelname)s - %(message)s'
    coloredlogs.install(level='CRITICAL')
    coloredlogs.install(level='DEBUG', fmt=logger_format, logger=logger, stream=sys.stdout, isatty=True)

    data_path = st.secrets.get('data_path')
    st.set_page_config(layout='wide', page_icon='🏀')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # Suppressing pycharm warnings
        app(data_path)


if __name__ == '__main__':
    main()
