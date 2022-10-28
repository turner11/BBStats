import pytest
from fluentcheck import Check

from apps import app_utils

CSV_3_SNAPSHOTS = '''
#1,#2,#3,#4,#5,Points,Points Against,Quarter,Time Left
1, 2, 3, 4, 5, 0, 0, 1, 10:00
1, 2, 3, 4, 6, 1, 0, 2, 10:00
1, 2, 3, 4, 7, 2, 3, 4, 0:00
'''


google_sheets_url = "https://docs.google.com/spreadsheets/d/1xvlTs0ry_f-jg3iRM2wdiwM7v1YMiRC7oJwI6MDGpuU/edit?usp=sharing"


@pytest.mark.parametrize('data, expected_count',
                         [
                             (google_sheets_url, None),
                             (CSV_3_SNAPSHOTS, 3),
                         ],
                         ids=['google sheets', '3 records csv'], )
def test_parse_data(data, expected_count):
    df = app_utils.get_snapshots_df(data)
    df_raw = df[~df.auto_added]
    if expected_count is not None:
        Check(df_raw).is_not_none().is_nuple(expected_count)
    else:
        Check(df_raw).is_not_none().is_not_empty()

