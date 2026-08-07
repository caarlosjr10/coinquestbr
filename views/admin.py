"""
views/admin.py — Painel administrativo (protegido por senha): conceder/revogar
planos manualmente e visualizar leads capturados na página de Planos.

Enquanto não há gateway de pagamento integrado, esta é a forma de ativar o
plano de um cliente que pagou por fora (ex: Pix combinado diretamente).
"""

import streamlit as st

import subscriptions

st.title("🔑 Painel Administrativo")

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not ADMIN_PASSWORD:
    st.error(
        "ADMIN_PASSWORD não configurada em .streamlit/secrets.toml. "
        "Defina uma senha para proteger este painel."
    )
    st.stop()

if not st.session_state.admin_authenticated:
    with st.form("admin_login"):
        password = st.text_input("Senha de administrador", type="password")
        if st.form_submit_button("Entrar"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Senha incorreta.")
    st.stop()

if st.button("Sair"):
    st.session_state.admin_authenticated = False
    st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Conceder / revogar planos
# ---------------------------------------------------------------------------

st.markdown("### Conceder plano a um cliente")
with st.form("grant_form"):
    col1, col2 = st.columns(2)
    grant_email = col1.text_input("E-mail do cliente")
    grant_plan_choice = col2.selectbox("Plano", ["Pro", "VIP"])
    grant_note = st.text_input("Observação (opcional)", placeholder="Ex: Pix recebido em 07/08 - referência #123")

    if st.form_submit_button("Conceder plano", type="primary"):
        if not grant_email or "@" not in grant_email:
            st.error("Informe um e-mail válido.")
        else:
            subscriptions.grant_plan(grant_email, grant_plan_choice, grant_note)
            st.success(f"Plano {grant_plan_choice} concedido para {grant_email}.")
            st.rerun()

st.markdown("### Assinaturas ativas")
active_subs = subscriptions.list_subscriptions()
if not active_subs:
    st.caption("Nenhuma assinatura ativa ainda.")
else:
    subs_table = [
        {"E-mail": email, "Plano": data["plan"], "Concedido em": data["granted_at"], "Observação": data.get("note", "")}
        for email, data in active_subs.items()
    ]
    st.dataframe(subs_table, use_container_width=True)

    revoke_email = st.selectbox("Revogar assinatura de:", [""] + list(active_subs.keys()))
    if revoke_email and st.button("🗑️ Revogar (volta para Grátis)"):
        subscriptions.revoke_plan(revoke_email)
        st.success(f"Assinatura de {revoke_email} revogada.")
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Leads capturados na página de Planos
# ---------------------------------------------------------------------------

st.markdown("### Leads (interesse de compra) capturados")
leads = subscriptions.list_leads()
if not leads:
    st.caption("Nenhum lead capturado ainda.")
else:
    st.dataframe(leads, use_container_width=True)
