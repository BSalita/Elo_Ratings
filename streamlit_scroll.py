"""Scroll the Streamlit page to new output after a row click.

A row click reruns the app and adds grids below the one clicked, but the
browser keeps the old scroll position, so the new grids start out of view.
scroll_to_here() marks the point where the new output begins and scrolls it
to the top of the window once per token. Streamlit keeps a components.html
iframe whose HTML is unchanged across reruns, so the script runs again only
when the token changes, i.e. on a new click, not on sorts or other reruns.
"""

from __future__ import annotations

import hashlib

import streamlit as st
import streamlit.components.v1 as components


def scroll_to_here(token: str) -> None:
    """Put this point at the top of the window, once per distinct token."""
    anchor = "scroll-anchor-" + hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
    components.html(
        f"""<script>
(function () {{
  const doc = window.parent.document;
  let tries = 0;
  const go = () => {{
    const el = doc.getElementById("{anchor}");
    if (el) {{
      el.scrollIntoView({{ block: "start" }});
      return;
    }}
    if (++tries < 50) setTimeout(go, 100);
  }};
  go();
}})();
</script>""",
        height=0,
    )
