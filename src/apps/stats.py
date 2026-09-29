from urllib.parse import urldefrag

import pandas as pd
import streamlit as st

from bbstats import SORTS, get_snapshots_df, get_stats_from_raw_data, to_csv_url

st.set_page_config(layout='wide', page_icon='🏀', page_title='BBStats')


@st.cache_data(ttl=15)
def load_snapshots(url, minutes):
    return get_snapshots_df(url, minutes_in_quarter=minutes)


@st.cache_data(ttl=300)
def load_images(players_url):
    return pd.read_csv(to_csv_url(players_url)).set_index('player').image.dropna().to_dict()


def secret(key):
    try:
        return st.secrets.get(key, '')
    except Exception:  # no secrets.toml
        return ''


qp = st.query_params
with st.sidebar:
    url = st.text_input('Data URL', qp.get('data') or secret('data_path'))
    minutes = st.radio('Minutes in quarter', [10, 12], horizontal=True)
    size = st.segmented_control('Group size', [1, 2, 3, 4, 5], default=5) or 5
    sort = st.segmented_control('Sort', list(SORTS), default='top') or 'top'

if not url:
    st.info('Paste a Google Sheets URL (shared "anyone with the link") in the sidebar.')
    st.stop()
qp['data'] = url

df_stats = get_stats_from_raw_data(load_snapshots(url, minutes), size, sort)
st.dataframe(
    df_stats.drop(columns=['elapsed']), hide_index=True, width='stretch',
    column_config={c: st.column_config.NumberColumn(format='%.2f') for c in ('score_pm', 'offense_pm', 'defence_pm')},
)

players_sheet = qp.get('players') or secret('players_sheet_id')
images = {}
if players_sheet:
    try:
        images = load_images(f'{urldefrag(url)[0]}#gid={players_sheet}')
    except Exception as ex:
        st.warning(f'Failed to get player images: {ex}')

for _, row in df_stats.head(10).iterrows():
    st.subheader(f'{row.played} min, {row.offense_diff}:{row.defence_diff}')
    for col, p in zip(st.columns(len(row.players)), row.players):
        if p in images:
            col.image(images[p], caption=f'#{p}')
        else:
            col.metric('Player', f'#{p}')
