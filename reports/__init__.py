"""Exports and notifications: the weekly PDF report, CSV export, email and Slack."""

import logging


def quiet_streamlit() -> None:
    """ui.data's @st.cache_data wrappers log "No runtime found" when imported outside Streamlit (CLI,
    collector); that is expected here, so only errors from that logger are shown."""
    from streamlit.logger import get_logger

    get_logger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
