"""
branding.py — Identidade visual compartilhada entre as páginas.

Enquanto não há um arquivo de logo definitivo, usamos um wordmark estilizado
via HTML/CSS. Para trocar por uma logo de verdade (PNG/SVG), substitua o
conteúdo de LOGO_HTML por uma tag <img src="data:image/png;base64,...">, ou
troque render_sidebar_logo() por st.sidebar.image("assets/logo.png").
"""

import streamlit as st

LOGO_HTML = """
<div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
  <div style="font-size:26px; line-height:1;">📈</div>
  <div style="font-size:21px; font-weight:800; letter-spacing:-0.5px; color:#F0F2F6;">
    CoinQuest<span style="color:#2E86C1;">BR</span>
  </div>
</div>
"""


def render_sidebar_logo() -> None:
    st.sidebar.markdown(LOGO_HTML, unsafe_allow_html=True)
