"""
views/backtest.py — Página principal: descrição da estratégia (interpretada
por IA), backtest, dashboard de métricas, parecer técnico, exportação e
alertas via Telegram.
"""

from dataclasses import replace

import plotly.graph_objects as go
import streamlit as st

import ai_analysis
import branding
import rate_limiter
import reports
import strategy_parser
import subscriptions
from engine import (
    DEFAULT_TIMEFRAME,
    MARKET_CONFIG,
    SMCParams,
    StrategyParams,
    calculate_metrics,
    default_candle_count,
    describe_active_filters,
    describe_risk,
    describe_trigger,
    fetch_data,
    format_timedelta,
    get_current_signal,
    get_current_signal_smc,
    required_warmup_bars,
    required_warmup_bars_smc,
    resolve_symbol,
    run_backtest,
    run_backtest_smc,
    run_optimization,
    timeframe_options,
)
from integrations import telegram_bot

# Campos de gerenciamento de risco (SL/TP em % / pips / ATR + trailing stop) —
# compartilhados por QUALQUER família de estratégia (RiskCostParams em
# engine.py), extraídos pela IA do texto porque variam por estratégia.
# spread/slippage/comissão ficam de fora: são configuração de corretora/conta,
# não da estratégia, então viram campos fixos na barra lateral.
RISK_COST_PARAM_FIELDS = [
    "sl_tp_mode", "stop_loss_pips", "take_profit_pips",
    "atr_period", "atr_sl_multiplier", "atr_tp_multiplier",
    "use_trailing_stop", "trailing_activation_pct", "trailing_distance_pct",
]

# Campos de StrategyParams preenchidos pela IA (strategy_parser) e espelhados
# em st.session_state como "cfg_<campo>" — usados tanto para montar o
# StrategyParams quanto para os widgets do expander "Ajustar manualmente".
# stop_loss_pct/take_profit_pct ficam de fora — são compartilhados com a
# família SMC via "cfg_sl"/"cfg_tp".
CLASSIC_PARAM_FIELDS = [
    "entry_trigger", "ma_type", "fast_period", "slow_period", "bb_period", "bb_std",
    "donchian_period", "candle_pattern",
    "use_rsi_filter", "rsi_period", "rsi_oversold", "rsi_overbought",
    "use_macd_filter", "macd_fast", "macd_slow", "macd_signal",
    "use_volume_filter", "volume_period", "volume_multiplier",
    "use_stochastic_filter", "stoch_k_period", "stoch_d_period", "stoch_oversold", "stoch_overbought",
    "use_adx_filter", "adx_period", "adx_threshold",
    "use_williams_filter", "williams_period", "williams_oversold", "williams_overbought",
    "use_cci_filter", "cci_period", "cci_oversold", "cci_overbought",
    "use_triple_ma", "mid_period",
] + RISK_COST_PARAM_FIELDS

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
    "strategy_family": "classic",
    "cfg_symbol": "",
    "cfg_timeframe": "",
    "cfg_n_candles": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _classic_params_from_session(initial_capital=10000.0, position_size_pct=100.0, **cost_kwargs) -> StrategyParams:
    return StrategyParams(
        **{k: st.session_state[f"cfg_{k}"] for k in CLASSIC_PARAM_FIELDS},
        stop_loss_pct=st.session_state.cfg_sl, take_profit_pct=st.session_state.cfg_tp,
        initial_capital=initial_capital, position_size_pct=position_size_pct,
        **cost_kwargs,
    )


def _smc_params_from_session(initial_capital=10000.0, position_size_pct=100.0, **cost_kwargs) -> SMCParams:
    return SMCParams(
        swing_strength=st.session_state.cfg_swing_strength,
        max_setup_bars=st.session_state.cfg_max_setup_bars,
        **{k: st.session_state[f"cfg_{k}"] for k in RISK_COST_PARAM_FIELDS},
        stop_loss_pct=st.session_state.cfg_sl, take_profit_pct=st.session_state.cfg_tp,
        initial_capital=initial_capital, position_size_pct=position_size_pct,
        **cost_kwargs,
    )


def _classic_desc(p: StrategyParams, suffix="") -> str:
    filters = describe_active_filters(p)
    filters_txt = f" + filtro {'/'.join(filters)}" if filters else ""
    return f"{describe_trigger(p)}{filters_txt}, {describe_risk(p)}{suffix}"


def _smc_desc(p: SMCParams, suffix="") -> str:
    return (
        f"Reversão SMC — varredura + BOS + pullback + CHoCH (swing={p.swing_strength}, "
        f"janela={p.max_setup_bars} candles), {describe_risk(p)}{suffix}"
    )


def _run_backtest_flow(family, market_type, symbol, timeframe, n_candles, initial_capital, position_size_pct,
                        run_mode, base_params, opt_ranges=None, max_combinations=None) -> bool:
    """Busca os dados e roda o backtest (único ou otimização), guardando o resultado na sessão.

    Retorna True se um backtest foi de fato executado (usado para contar o uso
    contra o limite mensal do plano), False se falhou antes de rodar.
    """
    try:
        df = fetch_data(market_type, symbol, timeframe, n_candles)
    except Exception as e:
        st.error(f"Erro ao buscar dados para '{symbol}': {e}")
        return False

    min_bars = required_warmup_bars(base_params) if family == "classic" else required_warmup_bars_smc(base_params)
    if df.empty or len(df) < min_bars:
        st.error("Dados insuficientes para essa configuração. Tente aumentar o número de candles.")
        return False

    if run_mode == "Backtest Único":
        if family == "smc":
            strategy_desc = _smc_desc(base_params)
            trades, equity_df = run_backtest_smc(df, base_params)
            current_signal = get_current_signal_smc(df, base_params)
        else:
            strategy_desc = _classic_desc(base_params)
            trades, equity_df = run_backtest(df, base_params)
            current_signal = get_current_signal(df, base_params)

        metrics = calculate_metrics(trades, equity_df, initial_capital)
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
            return False

        best = results_df.iloc[0]
        best_params = replace(
            base_params, fast_period=int(best["fast_period"]), slow_period=int(best["slow_period"]),
            stop_loss_pct=best["stop_loss_pct"], take_profit_pct=best["take_profit_pct"],
        )
        best_trades, best_equity_df = run_backtest(df, best_params)
        best_metrics = calculate_metrics(best_trades, best_equity_df, initial_capital)
        best_strategy_desc = _classic_desc(best_params, suffix=" (melhor combinação)")
        current_signal = get_current_signal(df, best_params)

        st.session_state.backtest_result = {
            "mode": "optimization", "metrics": best_metrics, "equity_df": best_equity_df,
            "trades": best_trades, "symbol": symbol, "strategy_desc": best_strategy_desc,
            "results_df": results_df, "current_signal": current_signal,
        }

    st.session_state.ai_report = None
    return True


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
st.sidebar.page_link("views/planos.py", label="Ver planos e assinar", icon="💳")

st.sidebar.subheader("2. Capital e Custos")
initial_capital = st.sidebar.number_input("Capital Inicial", min_value=100.0, value=10000.0, step=100.0)
position_size_pct = 100.0  # simplificado: sempre 100% do capital alocado por trade

with st.sidebar.expander("Custos operacionais"):
    st.caption("Configuração da sua corretora — descontada na abertura e no fechamento de cada trade.")
    spread_pct = st.number_input(
        "Spread (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.3f",
        help="Diferença entre preço de compra e venda do book.",
    )
    slippage_pct = st.number_input(
        "Slippage (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.3f",
        help="Derrapagem esperada na execução, além do spread.",
    )
    commission_pct = st.number_input(
        "Comissão da corretora (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.001, format="%.3f",
        help="Ex: 0.075% é a taxa taker padrão da Binance para Cripto.",
    )
    commission_fixed = st.number_input(
        "Custo fixo por operação (em pontos do preço)", min_value=0.0, value=0.0, step=0.0001, format="%.5f",
        help="Ex: um custo fixo de alguns pips numa corretora forex. Deixe 0 se não se aplica.",
    )
    pip_size = st.number_input(
        "Tamanho de 1 pip/ponto do ativo", min_value=0.000001, value=0.0001, step=0.0001, format="%.6f",
        help=(
            "Usado só quando o Stop Loss/Take Profit da estratégia estiver em modo "
            "'pips'. Padrão 0.0001 para a maioria dos pares forex — ajuste para "
            "0.01 em pares com JPY, ações ou índices."
        ),
    )
cost_kwargs = dict(
    spread_pct=spread_pct, slippage_pct=slippage_pct,
    commission_pct=commission_pct, commission_fixed=commission_fixed, pip_size=pip_size,
)

st.sidebar.subheader("3. Modo de Execução")
run_mode = st.sidebar.radio(
    "Modo",
    ["Testar como descrevi", "Otimizar automaticamente"],
    help=(
        "**Testar como descrevi:** roda o backtest exatamente com os parâmetros que a IA "
        "identificou na sua estratégia.\n\n"
        "**Otimizar automaticamente:** testa dezenas de pequenas variações da sua estratégia "
        "(outros períodos de média, outros stops/alvos) e te mostra a combinação com melhor "
        "resultado histórico."
    ),
)
run_mode = "Backtest Único" if run_mode == "Testar como descrevi" else "Otimização (Grid Search)"
if run_mode == "Otimização (Grid Search)":
    st.sidebar.caption("⚠️ Otimização automática só está disponível para estratégias de médias móveis/RSI.")

opt_fast_range = opt_slow_range = opt_sl_range = opt_tp_range = None
max_combinations = None
if run_mode == "Otimização (Grid Search)":
    st.sidebar.caption("Ajuste as faixas de valores que o sistema deve testar automaticamente:")
    opt_fast_range = st.sidebar.slider("Faixa Período Rápido", 2, 50, (5, 15))
    opt_slow_range = st.sidebar.slider("Faixa Período Lento", 10, 100, (20, 40))
    opt_sl_range = st.sidebar.slider("Faixa Stop Loss (%)", 0.5, 10.0, (1.0, 3.0))
    opt_tp_range = st.sidebar.slider("Faixa Take Profit (%)", 0.5, 20.0, (2.0, 6.0))
    max_combinations = st.sidebar.number_input("Máx. de combinações", min_value=10, max_value=500, value=100)

# --- Painel do Bot Privado do Telegram (exclusivo VIP) ---
st.sidebar.subheader("4. 🔔 Bot de Alertas do Telegram")
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

st.markdown("#### 1. Mercado e Timeframe")
col_market, col_tf = st.columns(2)
market_type = col_market.selectbox("Mercado", list(MARKET_CONFIG.keys()))

tf_options = timeframe_options(market_type)
tf_source = MARKET_CONFIG[market_type]["source"]
tf_default = DEFAULT_TIMEFRAME[tf_source]
tf_index = tf_options.index(tf_default) if tf_default in tf_options else 0
timeframe = col_tf.selectbox("Timeframe", tf_options, index=tf_index)

st.caption(f"{default_candle_count(timeframe)} candles históricos serão usados automaticamente para esse timeframe.")

st.markdown("#### 2. Descreva sua Estratégia")

strategy_description = st.text_area(
    "Descreva sua estratégia da forma mais precisa possível",
    height=120,
    placeholder=(
        "Exemplo: Comprar quando a média móvel exponencial de 9 períodos cruzar para cima da de "
        "21 períodos, com RSI abaixo de 35. Sair com stop loss de 2% ou take profit de 5%."
    ),
)

opt_ranges = (opt_fast_range, opt_slow_range, opt_sl_range, opt_tp_range) if run_mode != "Backtest Único" else None

bt_allowed, bt_used, bt_limit = (False, 0, None)
if user_email:
    bt_allowed, bt_used, bt_limit = rate_limiter.check_limit(user_email, user_plan, kind="backtest")
    st.caption(f"Backtests este mês: {bt_used}/{bt_limit} ({user_plan})")
    if not bt_allowed:
        if user_plan == "Grátis":
            upgrade_msg = "Faça upgrade para o plano **Pro** ou **VIP** para rodar muito mais backtests por mês."
        else:
            upgrade_msg = "Seu limite será renovado no próximo mês."
        st.error(f"⚠️ Você atingiu o limite de {bt_limit} backtests no plano **{user_plan}** este mês. {upgrade_msg}")
else:
    st.caption("Informe seu e-mail na barra lateral para rodar o backtest.")

if st.button("🚀 Analisar Estratégia", type="primary", disabled=not (user_email and bt_allowed)):
    with st.spinner("Interpretando sua estratégia e rodando o backtest..."):
        try:
            parsed = strategy_parser.parse_strategy_text(strategy_description)
            st.session_state.strategy_notes = parsed["interpretation_notes"]
            st.session_state.strategy_supported = parsed["is_supported"]

            if parsed["is_supported"]:
                family = parsed["strategy_family"]
                st.session_state.strategy_family = family
                st.session_state.strategy_interpreted = True

                symbol = resolve_symbol(market_type, parsed["symbol"])
                n_candles = default_candle_count(timeframe)
                st.session_state.cfg_symbol = symbol
                st.session_state.cfg_timeframe = timeframe
                st.session_state.cfg_n_candles = n_candles
                st.session_state.cfg_sl = parsed["stop_loss_pct"]
                st.session_state.cfg_tp = parsed["take_profit_pct"]

                effective_run_mode = run_mode
                if family == "smc":
                    st.session_state.cfg_swing_strength = parsed["swing_strength"]
                    st.session_state.cfg_max_setup_bars = parsed["max_setup_bars"]
                    for k in RISK_COST_PARAM_FIELDS:
                        st.session_state[f"cfg_{k}"] = parsed[k]
                    base_params = SMCParams(
                        swing_strength=parsed["swing_strength"], max_setup_bars=parsed["max_setup_bars"],
                        **{k: parsed[k] for k in RISK_COST_PARAM_FIELDS},
                        stop_loss_pct=parsed["stop_loss_pct"], take_profit_pct=parsed["take_profit_pct"],
                        initial_capital=initial_capital, position_size_pct=position_size_pct,
                        **cost_kwargs,
                    )
                    if run_mode != "Backtest Único":
                        st.info("Otimização automática ainda não está disponível para estratégias SMC — rodando o backtest único.")
                        effective_run_mode = "Backtest Único"
                else:
                    for k in CLASSIC_PARAM_FIELDS:
                        st.session_state[f"cfg_{k}"] = parsed[k]
                    base_params = StrategyParams(
                        **{k: parsed[k] for k in CLASSIC_PARAM_FIELDS},
                        stop_loss_pct=parsed["stop_loss_pct"], take_profit_pct=parsed["take_profit_pct"],
                        initial_capital=initial_capital, position_size_pct=position_size_pct,
                        **cost_kwargs,
                    )
                    unsupported_opt = base_params.entry_trigger != "ma_cross" or base_params.sl_tp_mode != "percent"
                    if unsupported_opt and run_mode != "Backtest Único":
                        st.info("Otimização automática só está disponível para o gatilho de cruzamento de médias com SL/TP percentual — rodando o backtest único.")
                        effective_run_mode = "Backtest Único"

                ran = _run_backtest_flow(
                    family, market_type, symbol, timeframe, n_candles, initial_capital, position_size_pct,
                    effective_run_mode, base_params, opt_ranges, max_combinations,
                )
                if ran:
                    rate_limiter.increment_usage(user_email, kind="backtest")
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
    if st.session_state.strategy_family == "smc":
        strategy_desc = _smc_desc(_smc_params_from_session())
    else:
        strategy_desc = _classic_desc(_classic_params_from_session())

    st.success(f"📋 **Estratégia interpretada:** {strategy_desc}")
    st.caption(
        f"Ativo: **{st.session_state.cfg_symbol}** · Timeframe: **{st.session_state.cfg_timeframe}** · "
        f"{st.session_state.cfg_n_candles} candles históricos"
    )
    if st.session_state.strategy_notes:
        st.caption(st.session_state.strategy_notes)

    with st.expander("⚙️ Ajustar parâmetros manualmente"):
        family = st.session_state.strategy_family

        if family == "smc":
            col_sw, col_win = st.columns(2)
            col_sw.number_input(
                "Força do swing (candles de cada lado)", min_value=1, max_value=10, key="cfg_swing_strength",
                help="Quantos candles de cada lado confirmam um topo/fundo. Valores maiores = estrutura mais suave, sinais mais raros.",
            )
            col_win.number_input(
                "Janela máx. do setup (candles)", min_value=5, max_value=200, key="cfg_max_setup_bars",
                help="Quantos candles esperar entre a varredura de liquidez e a confirmação de entrada antes de descartar o setup.",
            )
        else:
            trigger_labels = {
                "ma_cross": "Cruzamento de Médias Móveis", "bb_reversal": "Reversão nas Bandas de Bollinger",
                "donchian_breakout": "Rompimento de Canal (Donchian)", "candle_pattern": "Padrão de Candle",
            }
            trigger_keys = list(trigger_labels.keys())
            st.selectbox(
                "Gatilho de entrada", trigger_keys, format_func=lambda k: trigger_labels[k], key="cfg_entry_trigger",
            )

            if st.session_state.cfg_entry_trigger == "ma_cross":
                col1, col2 = st.columns(2)
                col1.selectbox("Tipo de Média Móvel", ["EMA", "SMA"], key="cfg_ma_type")
                col_fast, col_slow = st.columns(2)
                col_fast.number_input("Período Rápido", min_value=2, max_value=200, key="cfg_fast_period")
                col_slow.number_input("Período Lento", min_value=3, max_value=400, key="cfg_slow_period")
            elif st.session_state.cfg_entry_trigger == "bb_reversal":
                col_bp, col_bs = st.columns(2)
                col_bp.number_input("Período Bollinger", min_value=5, max_value=200, key="cfg_bb_period")
                col_bs.number_input("Desvio Padrão", min_value=0.5, max_value=5.0, step=0.1, key="cfg_bb_std")
            elif st.session_state.cfg_entry_trigger == "donchian_breakout":
                st.number_input("Período do Canal (candles)", min_value=5, max_value=200, key="cfg_donchian_period")
            else:
                st.selectbox(
                    "Padrão de Candle", ["bullish_engulfing", "hammer"],
                    format_func=lambda k: "Engolfo de Alta" if k == "bullish_engulfing" else "Martelo",
                    key="cfg_candle_pattern",
                )

            st.markdown("**Filtros adicionais** (combinados com o gatilho acima)")

            use_rsi_filter = st.checkbox("RSI", key="cfg_use_rsi_filter")
            if use_rsi_filter:
                st.number_input("Período RSI", min_value=2, max_value=100, key="cfg_rsi_period")
                col_os, col_ob = st.columns(2)
                col_os.number_input("RSI Sobrevenda", min_value=1, max_value=50, key="cfg_rsi_oversold")
                col_ob.number_input("RSI Sobrecompra", min_value=50, max_value=99, key="cfg_rsi_overbought")

            use_macd_filter = st.checkbox("MACD (linha acima do sinal)", key="cfg_use_macd_filter")
            if use_macd_filter:
                col_mf, col_ms, col_msig = st.columns(3)
                col_mf.number_input("MACD Rápido", min_value=2, max_value=100, key="cfg_macd_fast")
                col_ms.number_input("MACD Lento", min_value=3, max_value=200, key="cfg_macd_slow")
                col_msig.number_input("MACD Sinal", min_value=2, max_value=100, key="cfg_macd_signal")

            use_volume_filter = st.checkbox("Volume acima da média", key="cfg_use_volume_filter")
            if use_volume_filter:
                col_vp, col_vm = st.columns(2)
                col_vp.number_input("Período Volume", min_value=2, max_value=200, key="cfg_volume_period")
                col_vm.number_input("Multiplicador", min_value=1.0, max_value=10.0, step=0.1, key="cfg_volume_multiplier")

            use_stochastic_filter = st.checkbox("Estocástico", key="cfg_use_stochastic_filter")
            if use_stochastic_filter:
                col_sk, col_sd = st.columns(2)
                col_sk.number_input("Período %K", min_value=2, max_value=100, key="cfg_stoch_k_period")
                col_sd.number_input("Período %D", min_value=1, max_value=50, key="cfg_stoch_d_period")
                col_so, col_sb = st.columns(2)
                col_so.number_input("Sobrevenda", min_value=1, max_value=50, key="cfg_stoch_oversold")
                col_sb.number_input("Sobrecompra", min_value=50, max_value=99, key="cfg_stoch_overbought")

            use_adx_filter = st.checkbox("ADX (força de tendência)", key="cfg_use_adx_filter")
            if use_adx_filter:
                col_ap, col_at = st.columns(2)
                col_ap.number_input("Período ADX", min_value=2, max_value=100, key="cfg_adx_period")
                col_at.number_input("Limiar ADX", min_value=5.0, max_value=60.0, step=1.0, key="cfg_adx_threshold")

            use_williams_filter = st.checkbox("Williams %R", key="cfg_use_williams_filter")
            if use_williams_filter:
                st.number_input("Período Williams %R", min_value=2, max_value=100, key="cfg_williams_period")
                col_wo, col_wb = st.columns(2)
                col_wo.number_input("Sobrevenda", min_value=-100.0, max_value=-50.0, step=1.0, key="cfg_williams_oversold")
                col_wb.number_input("Sobrecompra", min_value=-50.0, max_value=0.0, step=1.0, key="cfg_williams_overbought")

            use_cci_filter = st.checkbox("CCI", key="cfg_use_cci_filter")
            if use_cci_filter:
                st.number_input("Período CCI", min_value=2, max_value=200, key="cfg_cci_period")
                col_co, col_cb = st.columns(2)
                col_co.number_input("Sobrevenda", min_value=-300.0, max_value=-50.0, step=10.0, key="cfg_cci_oversold")
                col_cb.number_input("Sobrecompra", min_value=50.0, max_value=300.0, step=10.0, key="cfg_cci_overbought")

            use_triple_ma = st.checkbox(
                "Alinhamento de 3 Médias Móveis", key="cfg_use_triple_ma",
                help="Exige Média Rápida > Média Intermediária > Média Lenta (ou o inverso p/ saída) antes de disparar o sinal.",
            )
            if use_triple_ma:
                st.number_input("Período da Média Intermediária", min_value=2, max_value=300, key="cfg_mid_period")

        st.markdown("**Gerenciamento de Stop Loss / Take Profit**")
        sl_tp_labels = {"percent": "Percentual (%)", "pips": "Pips / Pontos Fixos", "atr": "Múltiplo de ATR"}
        st.selectbox(
            "Modo de cálculo", list(sl_tp_labels.keys()), format_func=lambda k: sl_tp_labels[k], key="cfg_sl_tp_mode",
        )

        if st.session_state.cfg_sl_tp_mode == "pips":
            col_slp, col_tpp = st.columns(2)
            col_slp.number_input("Stop Loss (pips)", min_value=0.1, max_value=2000.0, step=1.0, key="cfg_stop_loss_pips")
            col_tpp.number_input("Take Profit (pips)", min_value=0.1, max_value=4000.0, step=1.0, key="cfg_take_profit_pips")
        elif st.session_state.cfg_sl_tp_mode == "atr":
            col_atp, col_atsl, col_attp = st.columns(3)
            col_atp.number_input("Período ATR", min_value=2, max_value=100, key="cfg_atr_period")
            col_atsl.number_input("Múlt. SL (x ATR)", min_value=0.1, max_value=10.0, step=0.1, key="cfg_atr_sl_multiplier")
            col_attp.number_input("Múlt. TP (x ATR)", min_value=0.1, max_value=20.0, step=0.1, key="cfg_atr_tp_multiplier")
        else:
            col_sl, col_tp = st.columns(2)
            col_sl.number_input("Stop Loss (%)", min_value=0.1, max_value=50.0, step=0.1, key="cfg_sl")
            col_tp.number_input("Take Profit (%)", min_value=0.1, max_value=100.0, step=0.1, key="cfg_tp")

        use_trailing_stop = st.checkbox("Trailing Stop (stop móvel)", key="cfg_use_trailing_stop")
        if use_trailing_stop:
            col_tra, col_trd = st.columns(2)
            col_tra.number_input(
                "Ativa após lucro de (%)", min_value=0.05, max_value=50.0, step=0.1, key="cfg_trailing_activation_pct",
            )
            col_trd.number_input(
                "Distância do preço (%)", min_value=0.05, max_value=50.0, step=0.1, key="cfg_trailing_distance_pct",
            )

        if not bt_allowed:
            st.caption(f"⚠️ Limite de {bt_limit} backtests do plano {user_plan} atingido este mês.")
        if st.button("🔄 Rodar novamente com esses parâmetros", disabled=not bt_allowed):
            if family == "smc":
                manual_params = _smc_params_from_session(initial_capital, position_size_pct, **cost_kwargs)
            else:
                manual_params = _classic_params_from_session(initial_capital, position_size_pct, **cost_kwargs)
            st.session_state.cfg_timeframe = timeframe
            st.session_state.cfg_n_candles = default_candle_count(timeframe)
            if family == "smc":
                effective_run_mode = "Backtest Único"
            elif manual_params.entry_trigger != "ma_cross" or manual_params.sl_tp_mode != "percent":
                effective_run_mode = "Backtest Único"
            else:
                effective_run_mode = run_mode
            with st.spinner("Rodando backtest..."):
                ran = _run_backtest_flow(
                    family, market_type, st.session_state.cfg_symbol, timeframe,
                    st.session_state.cfg_n_candles, initial_capital, position_size_pct,
                    effective_run_mode, manual_params, opt_ranges, max_combinations,
                )
                if ran:
                    rate_limiter.increment_usage(user_email, kind="backtest")
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

    m7, m8, m9, m10 = st.columns(4)
    m7.metric("Sharpe Ratio", metrics["sharpe_ratio"])
    m8.metric("Máx. Derrotas Seguidas", metrics["max_consecutive_losses"])
    m9.metric("Tempo Médio por Trade", format_timedelta(metrics["avg_trade_duration"]))
    m10.metric("Duração do Drawdown Máx.", format_timedelta(metrics["max_drawdown_duration"]))

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
        _signal_emoji = {"Compra": "🟢", "Venda (short)": "🔴", "Saída": "🟡"}
        emoji = _signal_emoji.get(current_signal["signal"], "⚪")
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
