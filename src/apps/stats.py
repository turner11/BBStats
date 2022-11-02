from __future__ import annotations

import dataclasses
import functools
from collections import defaultdict
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
import logging
import coloredlogs
import sys
from urllib.parse import urldefrag
from PIL import Image, ImageFont
from PIL import ImageDraw
import app_utils

logger = logging.getLogger(__name__)

PARAM_NAME_URL = 'data'
PARAM_PLAYERS_SHEET_ID = 'players'

no_image = 'https://media.istockphoto.com/vectors/basketball-player-standing-and-holding-ball-vector-silhouette-vector-id1299295749?k=20&m=1299295749&s=170667a&w=0&h=D8t1TTfMp_E-W7Vn-HBqRqpfsbCb4QxuR5lthFBq0fs='

try:
    font = ImageFont.truetype('arial.ttf', size=16)
except Exception:
    font = None

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

    players_images = defaultdict(lambda: no_image)
    if params.players_url:
        with st.spinner('getting players'):
            try:
                _images = app_utils.get_player_images(params.players_url)
                players_images.update(_images)
            except Exception as ex:
                st.warning(f'Failed to get images: {ex}')

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
        players = sorted(row.players)
        if players:
            url_by_player = {player: players_images[player] for player in row.players}
            _, col_image, _ = container.columns([1, 6, 1])
            image = get_concatenated_image(url_by_player, show_text=True, height=200)
            col_image.image(image)

        st.markdown('___')


def get_concatenated_image(image_urls: dict[str, str], show_text=True, height=None):
    images = tuple(url_to_image(url, str(txt) if show_text else '', height) for txt, url in image_urls.items())
    total_width = sum(im.width for im in images)
    max_height = max(im.height for im in images)
    dst = Image.new('RGB', (total_width, max_height))
    horizontal_offset = 0
    for image in images:
        dst.paste(image, (horizontal_offset, 0))
        horizontal_offset += image.width

    return dst


@functools.lru_cache(512)
def url_to_image(url, text='', height=None):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    if height:
        aspect_ratio = img.width / img.height
        width = int(height * aspect_ratio)
        img = img.resize((width, height))
    img = img.convert('RGB')

    if text:
        draw = ImageDraw.Draw(img)
        location = (img.width // 2, 0)
        draw.text(location, text, (0, 255, 0), font=font)
    return img


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
