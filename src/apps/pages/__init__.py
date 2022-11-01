import os
import sys


def patch_env():
    from pathlib import Path
    import streamlit as st
    pages_dir = Path(__file__).parent
    apps_dir = pages_dir.parent
    src_dir = apps_dir.parent

    for folder in (apps_dir, apps_dir, src_dir):
        if folder.exists() and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))

    debug_data = {
        'cwd': os.getcwd(),
        'sys.path': sys.path,
    }
    st.write(debug_data)


patch_env()
