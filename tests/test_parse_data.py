import math
from io import StringIO
import subprocess
import sys

import pandas as pd
import pytest

import bbstats
from bbstats import get_snapshots_df, get_stats_from_raw_data, http_url_or_none, players_by_jersey, to_csv_url

CSV_3_SNAPSHOTS = '''
#1,#2,#3,#4,#5,Points,Points Against,Quarter,Time Left
1, 2, 3, 4, 5, 0, 0, 1, 10:00
1, 2, 3, 4, 6, 1, 0, 2, 10:00
1, 2, 3, 4, 7, 2, 3, 4, 0:00
'''

google_sheets_url = "https://docs.google.com/spreadsheets/d/1xvlTs0ry_f-jg3iRM2wdiwM7v1YMiRC7oJwI6MDGpuU/edit?usp=sharing"


def test_parse_data():
    df = get_snapshots_df(CSV_3_SNAPSHOTS)
    assert len(df[~df.auto_added]) == 3


@pytest.mark.network
def test_parse_data_google_sheets():
    df = get_snapshots_df(google_sheets_url)
    assert not df[~df.auto_added].empty
    assert df.game_time_left.max() > 0


def test_to_csv_url():
    base = "https://docs.google.com/spreadsheets/d/ABC"
    assert to_csv_url(f"{base}/edit?usp=sharing#gid=123") == f"{base}/export?format=csv&gid=123"
    assert to_csv_url(f"{base}/edit?usp=sharing") == f"{base}/export?format=csv"
    # fragment gid wins over the query gid (players tab appended to a data-tab share link)
    assert to_csv_url(f"{base}/edit?gid=0#gid=77") == f"{base}/export?format=csv&gid=77"
    assert to_csv_url("https://example.com/x.csv") == "https://example.com/x.csv"


def test_players_by_jersey():
    players = [
        {"jersey_number": 7, "name": "Avi", "images": [{"url": "a"}, {"url": "b"}]},  # first image wins
        {"jersey_number": None, "name": "No number", "images": [{"url": "c"}]},  # no number
        {"jersey_number": 9, "name": "Dan", "images": []},  # no images
        {"jersey_number": 0, "name": "", "images": [{"url": "z"}]},  # 0 is a real jersey; empty name
        {"jersey_number": 5, "images": []},  # missing name
    ]
    assert players_by_jersey(players) == {
        7: {"name": "Avi", "image": "a"},
        9: {"name": "Dan", "image": None},
        0: {"name": None, "image": "z"},
        5: {"name": None, "image": None},
    }


@pytest.mark.parametrize("url, expected", [
    ("https://x.co/g/1", "https://x.co/g/1"),
    ("HTTP://x.co", "http://x.co"),
    ("  https://x.co ", "https://x.co"),
    ("javascript:alert(1)", None),
    ("JaVaScRiPt:alert(1)", None),
    ("java\tscript:alert(1)", None),
    ("data:text/html,x", None),
    ("/games/1", None),
    ("https://", None),
    ("", None),
    (None, None),
])
def test_http_url_or_none(url, expected):
    assert http_url_or_none(url) == expected


HEADERS = "#1,#2,#3,#4,#5,Points,Points Against,Quarter,Time Left\n"


@pytest.mark.parametrize("arg", [
    HEADERS,
    HEADERS + ",,,,,,,,\n,,,,,,,,\n",
    pd.read_csv(StringIO(HEADERS)),
])
def test_empty_sheet_loads(arg):
    assert get_snapshots_df(arg).empty


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_stats_group_sizes(size):
    df = get_stats_from_raw_data(get_snapshots_df(CSV_3_SNAPSHOTS), size)
    assert not df.empty
    assert all(len(p) == size for p in df.players)
    assert (df.elapsed > 0).all()
    assert all(math.isfinite(v) for v in df.score_pm)


def test_stats_sort_modes():
    snaps = get_snapshots_df(CSV_3_SNAPSHOTS)
    d = get_stats_from_raw_data(snaps, 1, 'defense').defence_pm.tolist()
    assert d == sorted(d)
    o = get_stats_from_raw_data(snaps, 1, 'offense').offense_pm.tolist()
    assert o == sorted(o, reverse=True)
    t = get_stats_from_raw_data(snaps, 1, 'top').score_pm.tolist()
    assert t == sorted(t, reverse=True)
    with pytest.raises(ValueError):
        get_stats_from_raw_data(snaps, 1, 'bogus')


def test_stats_needs_only_snapshot_columns():
    cols = ['players', 'elapsed', 'offense_diff', 'defence_diff']
    df = pd.DataFrame([([1, 2], 2.0, 4, 2), ([1, 3], 1.0, 0, 3)], columns=cols)
    out = get_stats_from_raw_data(df, 1)
    row = out[out.players.apply(lambda p: p == [1])].iloc[0]
    assert row.score_diff == 4 - 2 + 0 - 3 and row.elapsed == 3
    assert get_stats_from_raw_data(pd.DataFrame(columns=cols), 2).empty


def test_import_has_no_side_effects():
    code = "import bbstats, sys; assert 'streamlit' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
