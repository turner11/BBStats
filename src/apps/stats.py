import json
import urllib.request
from html import escape
from urllib.parse import urldefrag

import pandas as pd
import streamlit as st

from bbstats import SORTS, get_snapshots_df, get_stats_from_raw_data, http_url_or_none, players_by_jersey, to_csv_url

st.set_page_config(layout='centered', page_icon='🏀', page_title='BBStats')

CSS = '''<style>
.lineups { display: grid; grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr)); gap: .5rem; }
.lu { border: 1px solid rgba(127,127,127,.3); border-radius: .5rem; padding: .5rem .625rem; }
.lu-head { display: flex; align-items: baseline; gap: .5rem; }
.lu-head .rank { font-size: .75rem; font-weight: 700; opacity: .6; }
.lu-head .diff { font-size: 1.25rem; font-weight: 800; }
.lu-head .min { margin-inline-start: auto; font-size: .8rem; font-variant-numeric: tabular-nums; }
.win { color: #16a34a; } .loss { color: #dc2626; }
.lu-players { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .375rem; margin: .375rem 0; }
.lu-players .photo { width: 100%; aspect-ratio: 1; border-radius: 50%; object-fit: cover; object-position: center 20%; }
.lu-players div.photo { display: flex; align-items: center; justify-content: center; font-weight: 700;
  background: rgba(127,127,127,.18); }
.lu-players .num { text-align: center; font-weight: 700; font-size: .8rem; line-height: 1.2; }
.lu-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .25rem; text-align: center;
  font-variant-numeric: tabular-nums; }
.lu-stats b { display: block; font-size: .9rem; }
.lu-stats span { display: block; font-size: .65rem; opacity: .7; white-space: nowrap; }
[data-testid="stMain"] button, [data-testid="stMain"] a[data-testid^="stBaseLinkButton"] { min-height: 2.75rem; }
</style>'''


@st.cache_data(ttl=15, show_spinner='Reading the sheet…')
def load_snapshots(url, minutes):
    return get_snapshots_df(url, minutes_in_quarter=minutes)


@st.cache_data(ttl=300)
def load_images(players_url):
    images = pd.read_csv(to_csv_url(players_url)).set_index('player').image.dropna().to_dict()
    return {player: {'name': None, 'image': image} for player, image in images.items()}


@st.cache_data(ttl=300)
def load_team_players(api_url):
    with urllib.request.urlopen(api_url, timeout=15) as resp:
        return players_by_jersey(json.load(resp))


def secret(key):
    try:
        return st.secrets.get(key, '')
    except Exception:  # no secrets.toml
        return ''


def player_card(jersey, player):
    """One player as HTML. Names/URLs come from an external API: every value is escaped or URL-checked."""
    name = escape(player.get('name') or '')
    image = http_url_or_none(player.get('image'))
    label = f'#{jersey}'
    photo = (f'<img class="photo" src="{escape(image)}" alt="{name or label}" loading="lazy">' if image
             else f'<div class="photo">{label}</div>')
    return f'<div title="{name}">{photo}<div class="num">{label}</div></div>'


def lineup_card(rank, row, roster):
    """One lineup: rank, +/-, minutes, the five players, then the per-minute and points stats."""
    diff = int(row.score_diff)
    tone = 'win' if diff > 0 else 'loss' if diff < 0 else ''
    stats = ((f'{row.score_pm:+.2f}', '+/- per min'),
             (f'{int(row.offense_diff)} : {int(row.defence_diff)}', 'pts : opp'),
             (f'{row.offense_pm:.2f}', 'pts per min'), (f'{row.defence_pm:.2f}', 'opp per min'))
    return (f'<div class="lu"><div class="lu-head"><span class="rank">#{rank}</span>'
            f'<span class="diff {tone}">{diff:+d}</span><span class="min">{row.played} min</span></div>'
            f'<div class="lu-players">{"".join(player_card(j, roster.get(j, {})) for j in row.players)}</div>'
            f'<div class="lu-stats">{"".join(f"<div><b>{v}</b><span>{k}</span></div>" for v, k in stats)}</div></div>')


qp = st.query_params
st.html(CSS)
with st.sidebar:
    url = st.text_input('Data URL', qp.get('data') or secret('data_path'))
    minutes = st.radio('Minutes in quarter', [10, 12], horizontal=True)

return_url = http_url_or_none(qp.get('return_url'))
if return_url:
    st.link_button('Back to the game', return_url, icon=':material/arrow_back:')

title_col, refresh_col = st.columns([3, 2], vertical_alignment='center')
title_col.title('Lineups')
if refresh_col.button('Refresh', type='primary', icon=':material/refresh:', width='stretch'):
    load_snapshots.clear()

if not url:
    st.info('Paste a Google Sheets URL (shared "anyone with the link") in the sidebar.')
    st.stop()
qp['data'] = url

size = st.segmented_control('Group size', [1, 2, 3, 4, 5], default=5) or 5
sort = st.segmented_control('Sort', list(SORTS), default='top') or 'top'

try:
    snapshots = load_snapshots(url, minutes)
except Exception as ex:  # noqa: BLE001 - any load failure gets the same actionable message
    st.error("Couldn't read the sheet. Make sure it's shared as 'Anyone with the link can view'.")
    st.text(f'Details: {ex}')
    st.stop()

if snapshots.empty:
    st.info('No snapshots yet: stats appear after the first row is entered. Tap Refresh.')
    st.stop()

df_stats = get_stats_from_raw_data(snapshots, size, sort)
if df_stats.empty:
    st.info('No lineup has played time yet. Tap Refresh after the next snapshot.')
    st.stop()

team_api = http_url_or_none(qp.get('team_api'))  # wins over the sheet tab
players_sheet = qp.get('players') or secret('players_sheet_id')
roster = {}
if team_api or players_sheet:
    try:
        if team_api:
            roster = load_team_players(team_api)
        else:
            roster = load_images(f'{urldefrag(url)[0]}#gid={players_sheet}')
    except Exception as ex:
        st.warning(f'Failed to get player info: {ex}')

cards = (lineup_card(i + 1, row, roster) for i, row in enumerate(df_stats.head(10).itertuples()))
st.html(f'<div class="lineups">{"".join(cards)}</div>')

with st.expander('All lineups (table)'):
    st.dataframe(
        df_stats.drop(columns=['elapsed']), hide_index=True, width='stretch',
        column_config={c: st.column_config.NumberColumn(format='%.2f')
                       for c in ('score_pm', 'offense_pm', 'defence_pm')},
    )
