"""
views/backtest.py — Página principal: descrição da estratégia (interpretada
por IA), backtest, dashboard de métricas, parecer técnico, exportação e
alertas via Telegram.
"""

import plotly.graph_objects as go
import streamlit as st

import ai_analysis
import branding
import rate_limiter
import reports
import strategy_parser
import subscriptions
from engine import StrategyParams, calculate_metrics, fetch_data, get_current_signal, run_backtest, run_optimization
from integrations import telegram_bot

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

for key, default in {
    "backtest_result": None,
    "ai_report": None,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "strategy_interpreted": False,
    "strategy_supported": None,
    "strategy_notes": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _build_strategy_desc(ma_type, fast, slow, use_rsi, sl, tp, suffix="") -> str:
    return (
        f"Cruzamento {ma_type} {fast}/{slow}"
        f"{' + filtro RSI' if use_rsi else ''}, "
        f"SL {sl}% / TP {tp}%{suffix}"
    )


def _run_backtest_flow(market_type, symbol, timeframe, n_candles, initial_capital, position_size_pct,
                        run_mode, base_params, opt_ranges=None, max_combinations=None) -> None:
    """Busca os dados e roda o backtest (único ou otimização), guardando o resultado na sessão."""
    try:
        df = fetch_data(market_type, symbol, timeframe, n_candles)
    except Exception as e:
        st.error(f"Erro ao buscar dados para '{symbol}': {e}")
        return

    if df.empty or len(df) < base_params.slow_period + 5:
        st.error("Dados insuficientes para essa configuração. Tente aumentar o número de candles.")
        return

    strategy_desc = _build_strategy_desc(
        base_params.ma_type, base_params.fast_period, base_params.slow_period,
        base_params.use_rsi_filter, base_params.stop_loss_pct, base_params.take_profit_pct,
    )

    if run_mode == "Backtest Único":
        trades, equity_df = run_backtest(df, base_params)
        metrics = calculate_metrics(trades, equity_df, initial_capital)
        current_signal = get_current_signal(df, base_params)
        st.session_state.backtest_result = {
            "mode": "single", "metrics": metrics, "equity_df": equity_df,
            "trades": trades, "symbol": symbol, "strategy_desc": strategy_desc,
            "current_signal": current_signal,
        }
    else:
        opt_fast_range, opt_slow_range, opt_sl_range, opt_tp_range = opt_ranges
        results_df = run_optimization(
            df, base_params,
            fast_range=range(opt_fast_range[0], opt_fast_range[1] + 1),
            slow_range=range(opt_slow_range[0], opt_slow_range[1] + 1),
            sl_range=[opt_sl_range[0], (opt_sl_range[0] + opt_sl_range[1]) / 2, opt_sl_range[1]],
            tp_range=[opt_tp_range[0], (opt_tp_range[0] + opt_tp_range[1]) / 2, opt_tp_range[1]],
            max_combinations=max_combinations,
        )

        if results_df.empty:
            st.warning("Nenhuma combinação válida encontrada (verifique se período rápido < período lento).")
            return

        best = results_df.iloc[0]
        best_params = StrategyParams(
            ma_type=base_params.ma_type, fast_period=int(best["fast_period"]), slow_period=int(best["slow_period"]),
            use_rsi_filter=base_params.use_rsi_filter, rsi_period=base_params.rsi_period,
            rsi_oversold=base_params.rsi_oversold, rsi_overbought=base_params.rsi_overbought,
            stop_loss_pct=best["stop_loss_pct"], take_profit_pct=best["take_profit_pct"],
            initial_capital=initial_capital, position_size_pct=position_size_pct,
        )
        best_trades, best_equity_df = run_backtest(df, best_params)
        best_metrics = calculate_metrics(best_trades, best_equity_df, initial_capital)
        best_strategy_desc = _build_strategy_desc(
            base_params.ma_type, int(best["fast_period"]), int(best["slow_period"]), base_params.use_rsi_filter,
            f"{best['stop_loss_pct']:.1f}", f"{best['take_profit_pct']:.1f}", suffix=" (melhor combinação)",
        )
        current_signal = get_current_signal(df, best_params)

        st.session_state.backtest_result = {
            "mode": "optimization", "metrics": best_metrics, "equity_df": best_equity_df,
            "trades": best_trades, "symbol": symbol, "strategy_desc": best_strategy_desc,
            "results_df": results_df, "current_signal": current_signal,
        }

    st.session_state.ai_report = None


# ---------------------------------------------------------------------------
# Sidebar — identificação, mercado e execução
# ---------------------------------------------------------------------------

branding.render_sidebar_logo()
st.sidebar.caption("Validação e backtest de estratégias de trading")

st.sidebar.subheader("1. Identificação")
user_email = st.sidebar.text_input("Seu e-mail", placeholder="voce@email.com")
user_plan = subscriptions.get_plan_for_email(user_email)
is_vip = user_plan == "VIP"

st.sidebar.info(f"Plano atual: **{user_plan}**")
st.sidebar.page_link("views/planos.py", label="💳 Ver planos e assinar", icon="💳")

st.sidebar.subheader("2. Mercado e Ativo")
market_type = st.sidebar.selectbox("Mercado", ["Cripto", "Ações / Forex / Índices"])

if market_type == "Cripto":
    symbol = st.sidebar.text_input("Ticker (ccxt)", value="BTC/USDT", help="Ex: BTC/USDT, ETH/USDT")
    timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
else:
    symbol = st.sidebar.text_input("Ticker (yfinance)", value="AAPL", help="Ex: AAPL, EURUSD=X, PETR4.SA, ^GSPC")
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "1d", "1wk"], index=1)

n_candles = st.sidebar.slider("Nº de candles históricos", min_value=500, max_value=5000, value=1500, step=100)

st.sidebar.subheader("3. Capital")
initial_capital = st.sidebar.number_input("Capital Inicial", min_value=100.0, value=10000.0, step=100.0)
position_size_pct = st.sidebar.slider("% do capital por trade", min_value=1, max_value=100, value=100)

st.sidebar.subheader("4. Modo de Execução")
run_mode = st.sidebar.radio("Modo", ["Backtest Único", "Otimização (Grid Search)"])

opt_fast_range = opt_slow_range = opt_sl_range = opt_tp_range = None
max_combinations = None
if run_mode == "Otimização (Grid Search)":
    st.sidebar.caption("Testa várias combinações de período rápido/lento e SL/TP, mantendo o filtro de RSI da estratégia interpretada.")
    opt_fast_range = st.sidebar.slider("Faixa Período Rápido", 2, 50, (5, 15))
    opt_slow_range = st.sidebar.slider("Faixa Período Lento", 10, 100, (20, 40))
    opt_sl_range = st.sidebar.slider("Faixa Stop Loss (%)", 0.5, 10.0, (1.0, 3.0))
    opt_tp_range = st.sidebar.slider("Faixa Take Profit (%)", 0.5, 20.0, (2.0, 6.0))
    max_combinations = st.sidebar.number_input("Máx. de combinações", min_value=10, max_value=500, value=100)

# --- Painel do Bot Privado do Telegram (exclusivo VIP) ---
st.sidebar.subheader("5. 🔔 Bot de Alertas do Telegram")
if not is_vip:
    st.sidebar.caption("Disponível apenas no plano VIP / Institutional.")
else:
    st.sidebar.caption("Configure seu bot privado (veja o guia no README).")
    st.session_state.telegram_bot_token = st.sidebar.text_input(
        "Bot Token", value=st.session_state.telegram_bot_token, type="password",
        placeholder="123456789:ABCdefGhIJKlmNoPQRstuVwxyZ",
    )
    st.session_state.telegram_chat_id = st.sidebar.text_input(
        "Chat ID", value=st.session_state.telegram_chat_id, placeholder="123456789",
    )
    if st.sidebar.button("📨 Enviar Alerta de Teste", use_container_width=True):
        ok, msg = telegram_bot.send_test_alert(
            st.session_state.telegram_bot_token, st.session_state.telegram_chat_id
        )
        (st.sidebar.success if ok else st.sidebar.error)(msg)


# ---------------------------------------------------------------------------
# Área principal — 1. Descrição da estratégia + backtest num clique só
# ---------------------------------------------------------------------------

st.title("📈 CoinQuestBR — Validação de Estratégias")

st.markdown("#### 1. Descreva sua Estratégia")
st.caption(
    f"Escreva do seu jeito, como explicaria pra um analista. A IA interpreta e já roda o "
    f"backtest sobre os últimos {n_candles} candles de {symbol}, trazendo o resultado completo. "
    "Suporte atual: cruzamento de médias móveis (SMA/EMA), filtro de RSI, Stop Loss e Take Profit."
)

strategy_description = st.text_area(
    "Sua estratégia",
    height=120,
    placeholder=(
        "Exemplo: Comprar quando a média móvel exponencial de 9 períodos cruzar para cima da "
        "de 21 períodos, com RSI abaixo de 35. Sair com stop loss de 2% ou take profit de 5%."
    ),
    label_visibility="collapsed",
)

opt_ranges = (opt_fast_range, opt_slow_range, opt_sl_range, opt_tp_range) if run_mode != "Backtest Único" else None

if st.button("🚀 Analisar Estratégia", type="primary"):
    with st.spinner("Interpretando sua estratégia e rodando o backtest..."):
        try:
            parsed = strategy_parser.parse_strategy_text(strategy_description)
            st.session_state.strategy_notes = parsed["interpretation_notes"]
            st.session_state.strategy_supported = parsed["is_supported"]

            if parsed["is_supported"]:
                st.session_state.cfg_ma_type = parsed["ma_type"]
                st.session_state.cfg_fast_period = parsed["fast_period"]
                st.session_state.cfg_slow_period = parsed["slow_period"]
                st.session_state.cfg_use_rsi = parsed["use_rsi_filter"]
                st.session_state.cfg_rsi_period = parsed["rsi_period"]
                st.session_state.cfg_rsi_oversold = parsed["rsi_oversold"]
                st.session_state.cfg_rsi_overbought = parsed["rsi_overbought"]
                st.session_state.cfg_sl = parsed["stop_loss_pct"]
                st.session_state.cfg_tp = parsed["take_profit_pct"]
                st.session_state.strategy_interpreted = True

                base_params = StrategyParams(
                    ma_type=parsed["ma_type"], fast_period=parsed["fast_period"], slow_period=parsed["slow_period"],
                    use_rsi_filter=parsed["use_rsi_filter"], rsi_period=parsed["rsi_period"],
                    rsi_oversold=parsed["rsi_oversold"], rsi_overbought=parsed["rsi_overbought"],
                    stop_loss_pct=parsed["stop_loss_pct"], take_profit_pct=parsed["take_profit_pct"],
                    initial_capital=initial_capital, position_size_pct=position_size_pct,
                )
                _run_backtest_flow(
                    market_type, symbol, timeframe, n_candles, initial_capital, position_size_pct,
                    run_mode, base_params, opt_ranges, max_combinations,
                )
            else:
                st.session_state.strategy_interpreted = False
                st.session_state.backtest_result = None
        except Exception as e:
            st.error(f"Erro ao interpretar estratégia: {e}")
            st.session_state.strategy_interpreted = False
            st.session_state.strategy_supported = None

# Estratégia descrita, mas a IA não conseguiu mapeá-la para o que o motor suporta
if st.session_state.strategy_supported is False:
    st.error(f"⚠️ {st.session_state.strategy_notes}")

if st.session_state.strategy_interpreted:
    strategy_desc = _build_strategy_desc(
        st.session_state.cfg_ma_type, st.session_state.cfg_fast_period, st.session_state.cfg_slow_period,
        st.session_state.cfg_use_rsi, st.session_state.cfg_sl, st.session_state.cfg_tp,
    )

    st.success(f"📋 **Estratégia interpretada:** {strategy_desc}")
    if st.session_state.strategy_notes:
        st.caption(st.session_state.strategy_notes)

    with st.expander("⚙️ Ajustar parâmetros manualmente"):
        col1, col2 = st.columns(2)
        col1.selectbox("Tipo de Média Móvel", ["EMA", "SMA"], key="cfg_ma_type")
        col_fast, col_slow = st.columns(2)
        col_fast.number_input("Período Rápido", min_value=2, max_value=200, key="cfg_fast_period")
        col_slow.number_input("Período Lento", min_value=3, max_value=400, key="cfg_slow_period")

        use_rsi_filter = st.checkbox("Usar filtro de RSI", key="cfg_use_rsi")
        if use_rsi_filter:
            st.number_input("Período RSI", min_value=2, max_value=100, key="cfg_rsi_period")
            col_os, col_ob = st.columns(2)
            col_os.number_input("RSI Sobrevenda", min_value=1, max_value=50, key="cfg_rsi_oversold")
            col_ob.number_input("RSI Sobrecompra", min_value=50, max_value=99, key="cfg_rsi_overbought")

        col_sl, col_tp = st.columns(2)
        col_sl.number_input("Stop Loss (%)", min_value=0.1, max_value=50.0, step=0.1, key="cfg_sl")
        col_tp.number_input("Take Profit (%)", min_value=0.1, max_value=100.0, step=0.1, key="cfg_tp")

        if st.button("🔄 Rodar novamente com esses parâmetros"):
            manual_params = StrategyParams(
                ma_type=st.session_state.cfg_ma_type,
                fast_period=st.session_state.cfg_fast_period,
                slow_period=st.session_state.cfg_slow_period,
                use_rsi_filter=st.session_state.cfg_use_rsi,
                rsi_period=st.session_state.cfg_rsi_period,
                rsi_oversold=st.session_state.cfg_rsi_oversold,
                rsi_overbought=st.session_state.cfg_rsi_overbought,
                stop_loss_pct=st.session_state.cfg_sl,
                take_profit_pct=st.session_state.cfg_tp,
                initial_capital=initial_capital, position_size_pct=position_size_pct,
            )
            with st.spinner("Rodando backtest..."):
                _run_backtest_flow(
                    market_type, symbol, timeframe, n_candles, initial_capital, position_size_pct,
                    run_mode, manual_params, opt_ranges, max_combinations,
                )
            st.rerun()


# ---------------------------------------------------------------------------
# Exibição dos resultados
# ---------------------------------------------------------------------------

result = st.session_state.backtest_result

if result is None:
    st.info("Descreva sua estratégia acima e clique em **🚀 Analisar Estratégia** para ver o resultado do backtest.")
else:
    metrics = result["metrics"]

    st.markdown("---")
    if result["mode"] == "optimization":
        st.subheader("🏆 Melhor Combinação Encontrada")
        st.caption(result["strategy_desc"])
        with st.expander("Ver ranking completo das combinações testadas"):
            st.dataframe(result["results_df"], use_container_width=True)
    else:
        st.subheader(f"Resultados — {result['symbol']}")
        st.caption(result["strategy_desc"])

    st.markdown("#### Dashboard de Métricas")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Win Rate", f"{metrics['win_rate']}%")
    m2.metric("Profit Factor", metrics["profit_factor"])
    m3.metric("Drawdown Máximo", f"{metrics['max_drawdown_pct']}%")
    m4.metric("Payoff", metrics["payoff"])
    m5.metric("Expectativa", metrics["expectancy"])
    m6.metric("Resultado Líquido", f"{metrics['net_profit_pct']}%")

    st.markdown(f"**Total de Trades:** {metrics['total_trades']} &nbsp;|&nbsp; "
                f"**Capital Final:** {metrics['final_equity']:,.2f}")

    st.markdown("#### Curva de Patrimônio (Equity Curve)")
    equity_df = result["equity_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["equity"],
        mode="lines", name="Patrimônio", line=dict(color="#2E86C1", width=2),
        fill="tozeroy", fillcolor="rgba(46, 134, 193, 0.1)",
    ))
    fig.update_layout(
        xaxis_title="Data", yaxis_title="Patrimônio",
        template="plotly_white", height=450, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if result["trades"]:
        with st.expander("Ver histórico de trades"):
            trades_display = [{
                "Entrada": t.entry_time, "Preço Entrada": round(t.entry_price, 4),
                "Saída": t.exit_time, "Preço Saída": round(t.exit_price, 4) if t.exit_price else None,
                "Motivo": t.exit_reason, "PnL": round(t.pnl, 2), "PnL %": round(t.pnl_pct, 2),
            } for t in result["trades"]]
            st.dataframe(trades_display, use_container_width=True)

    # -----------------------------------------------------------------
    # Sinal atual e alerta via Telegram (VIP)
    # -----------------------------------------------------------------

    current_signal = result.get("current_signal") or {}
    if current_signal.get("signal"):
        emoji = "🟢" if current_signal["signal"] == "Compra" else "🔴"
        st.markdown("---")
        st.markdown(f"#### {emoji} Sinal Ativo no Último Candle: {current_signal['signal']}")
        st.caption(
            f"Preço: {current_signal['price']:.4f} — {current_signal['reason']} — "
            f"{current_signal['timestamp']}"
        )

        if is_vip:
            if st.session_state.telegram_bot_token and st.session_state.telegram_chat_id:
                if st.button("📨 Disparar Alerta desse Sinal no Telegram"):
                    ok, msg = telegram_bot.send_signal_alert(
                        st.session_state.telegram_bot_token,
                        st.session_state.telegram_chat_id,
                        asset=result["symbol"],
                        signal_type=current_signal["signal"],
                        price=current_signal["price"],
                        reason=current_signal["reason"],
                    )
                    (st.success if ok else st.error)(msg)
            else:
                st.caption("Configure seu Bot Token e Chat ID na barra lateral para enviar este alerta.")

    # -----------------------------------------------------------------
    # Parecer Técnico com IA
    # -----------------------------------------------------------------

    st.markdown("---")
    st.markdown("#### 🤖 Parecer Técnico com IA")

    if not user_email:
        st.warning("Informe seu e-mail na barra lateral para gerar o parecer com IA.")
    else:
        allowed, used, limit = rate_limiter.check_limit(user_email, user_plan)
        st.caption(f"Uso este mês: {used}/{limit} análises ({user_plan})")

        gen_button = st.button("Gerar Parecer Técnico com IA", disabled=not allowed)

        if not allowed:
            if user_plan == "Grátis":
                upgrade_msg = "Faça upgrade para o plano **Pro** (50 análises/mês) ou **VIP** (500 análises/mês + recursos exclusivos)!"
            elif user_plan == "Pro":
                upgrade_msg = "Faça upgrade para o plano **VIP / Institutional** e tenha até 500 análises/mês, prioridade e exportação de relatórios!"
            else:
                upgrade_msg = "Seu limite VIP será renovado no próximo mês."
            st.error(f"⚠️ Você atingiu o limite de {limit} análise(s) de IA no plano **{user_plan}** este mês. {upgrade_msg}")

        if gen_button and allowed:
            with st.spinner("Consultando a IA para gerar o parecer técnico..."):
                try:
                    report = ai_analysis.generate_analysis(metrics, result["strategy_desc"], result["symbol"], plan=user_plan)
                    rate_limiter.increment_usage(user_email)
                    st.session_state.ai_report = report
                except Exception as e:
                    st.error(f"Erro ao gerar parecer: {e}")

        if st.session_state.ai_report:
            st.success(st.session_state.ai_report)

    # -----------------------------------------------------------------
    # Exportação de Relatório (exclusivo VIP)
    # -----------------------------------------------------------------

    st.markdown("---")
    st.markdown("#### 📄 Exportar Relatório Completo")

    if not is_vip:
        st.info("A exportação de relatórios em HTML/PDF é exclusiva do plano **VIP / Institutional**.")
    else:
        col_html, col_pdf = st.columns(2)

        html_report = reports.generate_html_report(
            result["symbol"], result["strategy_desc"], metrics, equity_df, st.session_state.ai_report,
        )
        col_html.download_button(
            "⬇️ Baixar Relatório (HTML)", data=html_report,
            file_name=f"coinquestbr_{result['symbol'].replace('/', '-')}.html",
            mime="text/html", use_container_width=True,
        )

        if col_pdf.button("🖨️ Gerar Relatório (PDF)", use_container_width=True):
            with st.spinner("Gerando PDF..."):
                try:
                    pdf_bytes = reports.generate_pdf_report(
                        result["symbol"], result["strategy_desc"], metrics, equity_df, st.session_state.ai_report,
                    )
                    st.download_button(
                        "⬇️ Baixar Relatório (PDF)", data=pdf_bytes,
                        file_name=f"coinquestbr_{result['symbol'].replace('/', '-')}.pdf",
                        mime="application/pdf", use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {e}")
