"""
views/planos.py — Página de Planos: apresentação comercial dos pacotes e
captura de interesse de compra (checkout ainda sem gateway de pagamento
integrado — ver views/admin.py para ativação manual).
"""

import streamlit as st

import branding
import subscriptions

branding.render_sidebar_logo()

st.markdown(
    """
    <style>
    div[class*="st-key-plan_card_"] {
        background: linear-gradient(180deg, #171e2c 0%, #10141f 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 8px 22px 22px 22px;
        position: relative;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    div[class*="st-key-plan_card_"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 30px rgba(0,0,0,0.35);
    }
    div[class*="st-key-plan_card_Pro"] {
        border: 1px solid #2E86C1;
        box-shadow: 0 0 0 1px rgba(46,134,193,0.25), 0 14px 30px rgba(46,134,193,0.12);
    }
    .plan-badge {
        position: absolute; top: -13px; left: 50%; transform: translateX(-50%);
        background: #2E86C1; color: white; font-size: 12px; font-weight: 700;
        padding: 4px 14px; border-radius: 999px; letter-spacing: 0.4px;
        white-space: nowrap;
    }
    .plan-icon { font-size: 30px; margin-top: 22px; margin-bottom: 4px; }
    .plan-name { font-size: 22px; font-weight: 800; color: #F0F2F6; margin-bottom: 2px; }
    .plan-tagline { color: #8b94a3; font-size: 13px; margin-bottom: 16px; min-height: 34px; }
    .plan-price-row { display:flex; align-items:baseline; gap:6px; margin-bottom: 18px; }
    .plan-price { font-size: 38px; font-weight: 800; color: #ffffff; }
    .plan-period { font-size: 14px; color: #8b94a3; font-weight: 500; }
    .plan-feature { display:flex; align-items:flex-start; gap:8px; margin: 9px 0; font-size:14px; color:#d7dbe3; line-height:1.4; }
    .plan-check { color:#3ddc84; flex-shrink:0; }
    div[class*="st-key-plan_card_"] div[data-testid="stButton"] button {
        width: 100%; border-radius: 10px; padding: 10px 0; font-weight: 700; margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💳 Planos CoinQuestBR")
st.caption("Escolha o plano ideal para o seu volume de análises e recursos.")

if "checkout_plan" not in st.session_state:
    st.session_state.checkout_plan = None

PLANS = [
    {
        "key": "Grátis",
        "icon": "🆓",
        "price": "R$ 0",
        "period": "/sempre",
        "tagline": "Para conhecer a plataforma",
        "features": [
            "Backtest ilimitado (Cripto, Ações, Forex, Índices)",
            "Estratégia interpretada por IA a partir de texto livre",
            "Dashboard completo de métricas + Equity Curve",
            "1 Parecer Técnico com IA por mês",
        ],
        "cta": "Plano Atual",
        "badge": None,
    },
    {
        "key": "Pro",
        "icon": "⚡",
        "price": "R$ 97",
        "period": "/mês",
        "tagline": "Para quem testa estratégias com frequência",
        "features": [
            "Tudo do plano Grátis",
            "50 Pareceres Técnicos com IA por mês",
            "Backtests ilimitados e otimização (Grid Search)",
        ],
        "cta": "Assinar Pro",
        "badge": "MAIS POPULAR",
    },
    {
        "key": "VIP",
        "icon": "👑",
        "price": "R$ 297",
        "period": "/mês",
        "tagline": "Para traders e gestoras com sinais em tempo real",
        "features": [
            "Tudo do plano Pro",
            "500 Pareceres Técnicos com IA por mês",
            "IA mais avançada — análise mais aprofundada",
            "Exportação de relatório completo (HTML e PDF)",
            "Bot privado de alertas no Telegram",
        ],
        "cta": "Assinar VIP",
        "badge": None,
    },
]

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
cols = st.columns(3, gap="medium")

for col, plan in zip(cols, PLANS):
    with col:
        with st.container(key=f"plan_card_{plan['key']}"):
            if plan["badge"]:
                st.markdown(f"<div class='plan-badge'>{plan['badge']}</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="plan-icon">{plan['icon']}</div>
                <div class="plan-name">{plan['key']}</div>
                <div class="plan-tagline">{plan['tagline']}</div>
                <div class="plan-price-row">
                    <span class="plan-price">{plan['price']}</span>
                    <span class="plan-period">{plan['period']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            features_html = "".join(
                f'<div class="plan-feature"><span class="plan-check">✓</span><span>{f}</span></div>'
                for f in plan["features"]
            )
            st.markdown(features_html, unsafe_allow_html=True)

            if plan["key"] == "Grátis":
                st.button(plan["cta"], key=f"cta_{plan['key']}", disabled=True)
            else:
                if st.button(plan["cta"], key=f"cta_{plan['key']}", type="primary"):
                    st.session_state.checkout_plan = plan["key"]
                    st.rerun()

# ---------------------------------------------------------------------------
# Checkout (placeholder — sem gateway de pagamento integrado ainda)
# ---------------------------------------------------------------------------

if st.session_state.checkout_plan:
    plan_key = st.session_state.checkout_plan
    plan_data = next(p for p in PLANS if p["key"] == plan_key)

    st.markdown("---")
    st.markdown(f"### 🧾 Assinar plano {plan_key} — {plan_data['price']}{plan_data['period']}")

    st.warning(
        "⚠️ O pagamento online ainda está sendo configurado. Deixe seu e-mail abaixo que "
        "avisaremos assim que a assinatura estiver disponível — ou entre em contato "
        "diretamente para ativação manual."
    )

    with st.form("checkout_form"):
        lead_email = st.text_input("Seu e-mail", placeholder="voce@email.com")
        submitted = st.form_submit_button("Quero ser avisado(a)", type="primary")

        if submitted:
            if not lead_email or "@" not in lead_email:
                st.error("Informe um e-mail válido.")
            else:
                subscriptions.add_lead(lead_email, plan_key)
                st.success("Interesse registrado! Avisaremos assim que o pagamento estiver liberado.")

    if st.button("← Voltar aos planos"):
        st.session_state.checkout_plan = None
        st.rerun()

st.markdown("---")
st.caption(
    "Já pagou por outro meio (ex: Pix combinado diretamente)? Seu plano será ativado "
    "manualmente pelo administrador em até algumas horas após a confirmação."
)
