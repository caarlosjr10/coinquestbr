"""
app.py — CoinQuestBR: ponto de entrada / roteador do app multi-página.

Páginas:
- views/backtest.py — estratégia por texto (IA), backtest, parecer técnico,
  exportação e alertas via Telegram.
- views/planos.py — apresentação comercial dos planos e captura de leads.
- views/admin.py — painel administrativo (protegido por senha) para conceder
  planos manualmente enquanto não há gateway de pagamento integrado.
"""

import streamlit as st

st.set_page_config(page_title="CoinQuestBR", page_icon="📈", layout="wide")

pages = [
    st.Page("views/backtest.py", title="Backtest & IA", icon="📊", default=True),
    st.Page("views/planos.py", title="Planos", icon="💳"),
]

# A página Admin fica OCULTA da navegação pública — só aparece se a URL tiver
# o token secreto configurado em ADMIN_URL_TOKEN (.streamlit/secrets.toml).
# Ex: https://seu-app.streamlit.app/admin?admin=SEU_TOKEN_SECRETO
# Dentro da página ainda há uma segunda camada: login por senha (ADMIN_PASSWORD).
admin_token = st.secrets.get("ADMIN_URL_TOKEN")
if admin_token and st.query_params.get("admin") == admin_token:
    pages.append(st.Page("views/admin.py", title="Admin", icon="🔑"))

pg = st.navigation(pages)
pg.run()
