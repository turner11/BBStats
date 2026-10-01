# BBStats

Basketball lineup +/- stats from a per-snapshot Google Sheet. `bbstats` is an importable, side-effect-free package.

    pip install -e .[app,dev]
    streamlit run src/apps/stats.py
    pytest                    # or: pytest -m "not network"

Query params for the app: `data` (sheet URL), `players` (gid of a `player,image` tab), and `team_api` (a URL returning
the gush-ball players JSON, e.g. `https://<site>/api/teams/<id>/players`; wins over `players`).

```python
from bbstats import get_snapshots_df, get_stats_from_raw_data

snapshots = get_snapshots_df("https://docs.google.com/spreadsheets/d/<id>/edit")  # shared "anyone with the link"
top_fives = get_stats_from_raw_data(snapshots, group_size=5, sort="top")  # or "offense" / "defense"
print(top_fives.head())
```
