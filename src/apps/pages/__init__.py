def patch_env():
    from pathlib import Path
    import streamlit as st
    apps_dir = Path(__file__).parent
    st.write(apps_dir)



patch_env()