"""
engine.py — Motor de dados e backtest do CoinQuestBR.

Responsável por:
- Buscar dados OHLCV (cripto via ccxt, ações/forex/índices via yfinance)
- Calcular indicadores (SMA/EMA, RSI)
- Rodar o backtest com Stop Loss / Take Profit
- Calcular métricas de performance
- Rodar otimização (grid search) de parâmetros
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aquisição de dados
# ---------------------------------------------------------------------------

def fetch_crypto_data(symbol: str, timeframe: str = "1h", limit: int = 1000, exchange_id: str = "binance") -> pd.DataFrame:
    """Busca OHLCV de um par cripto via ccxt (ex: BTC/USDT).

    A maioria das exchanges limita ~1000 candles por requisição — para pedidos
    maiores (o slider vai até 5000), pagina múltiplas chamadas de fetch_ohlcv
    andando para trás no tempo, em vez de devolver silenciosamente menos
    candles do que o usuário pediu.
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    per_request = 1000
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - limit * timeframe_ms

    all_candles: list = []
    while len(all_candles) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=per_request)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + timeframe_ms
        if len(batch) < per_request:
            break  # não há mais histórico disponível

    if not all_candles:
        raise ValueError(f"Nenhum dado retornado pela exchange para '{symbol}'.")

    all_candles = all_candles[-limit:]

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_yfinance_data(symbol: str, interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """Busca OHLCV de ações/forex/índices via yfinance (ex: AAPL, EURUSD=X, PETR4.SA)."""
    import yfinance as yf

    period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d",
        "1d": "max", "1wk": "max", "1mo": "max",
    }
    period = period_map.get(interval, "2y")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para o ticker '{symbol}'.")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]

    if len(df) > limit:
        df = df.iloc[-limit:]

    return df


def fetch_data(market_type: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Ponto único de entrada para buscar dados, roteando por tipo de mercado."""
    config = MARKET_CONFIG[market_type]
    if config["source"] == "ccxt":
        return fetch_crypto_data(symbol, timeframe=timeframe, limit=limit)
    return fetch_yfinance_data(symbol, interval=timeframe, limit=limit)


# ---------------------------------------------------------------------------
# Categorias de mercado, timeframes e candles padrão
# ---------------------------------------------------------------------------
#
# A partir da simplificação do formulário, o usuário escolhe uma categoria
# ampla de mercado (Forex / Crypto / Índices / Stocks) e um timeframe no topo
# da página — o ativo específico (ticker exato) vem do texto da estratégia,
# interpretado pela IA em strategy_parser.py. Este bloco resolve o ativo
# padrão quando a IA não extrai um ativo explícito do texto.

MARKET_CONFIG = {
    "Forex": {"source": "yfinance", "default_symbol": "EURUSD=X"},
    "Crypto": {"source": "ccxt", "default_symbol": "BTC/USDT"},
    "Índices": {"source": "yfinance", "default_symbol": "^GSPC"},
    "Stocks": {"source": "yfinance", "default_symbol": "AAPL"},
}

VALID_TIMEFRAMES = {
    "yfinance": ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
    "ccxt": ["1m", "5m", "15m", "1h", "4h", "1d"],
}

DEFAULT_TIMEFRAME = {"yfinance": "1d", "ccxt": "1h"}

# Nº de candles padrão por timeframe. Escolhido para aproximar o piso
# recomendado de amostra estatisticamente relevante em backtests (na prática,
# 200-500 trades executados é considerado confiável; abaixo de 100 trades os
# resultados tendem a ser ruído) sem estourar os limites de histórico
# disponíveis nas fontes de dados gratuitas (ex: yfinance intraday).
DEFAULT_CANDLES_BY_TIMEFRAME = {
    "1m": 1500,   # yfinance só guarda ~7 dias de histórico em 1m
    "5m": 2500,   # yfinance só guarda ~60 dias de histórico em 5m
    "15m": 3000,
    "30m": 2500,
    "1h": 2000,
    "4h": 1500,
    "1d": 750,
    "1wk": 260,
}
DEFAULT_CANDLES_FALLBACK = 1500


def timeframe_options(market_type: str) -> list[str]:
    """Timeframes válidos para a fonte de dados da categoria de mercado escolhida."""
    return VALID_TIMEFRAMES[MARKET_CONFIG[market_type]["source"]]


def resolve_symbol(market_type: str, ai_symbol: str) -> str:
    """Decide o ativo real a usar, a partir do que a IA extraiu do texto da estratégia.

    Se a IA não identificou um ativo específico (ou identificou algo incompatível
    com a categoria de mercado escolhida — ex: um par cripto com "/" numa
    categoria que usa yfinance), cai para o ativo padrão daquela categoria.
    """
    config = MARKET_CONFIG[market_type]
    source = config["source"]

    symbol = (ai_symbol or "").strip()
    looks_like_pair = "/" in symbol
    if source == "ccxt":
        if not symbol or not looks_like_pair:
            symbol = config["default_symbol"]
    else:
        if not symbol or looks_like_pair:
            symbol = config["default_symbol"]

    return symbol


def default_candle_count(timeframe: str) -> int:
    return DEFAULT_CANDLES_BY_TIMEFRAME.get(timeframe, DEFAULT_CANDLES_FALLBACK)


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


# ---------------------------------------------------------------------------
# Configuração da estratégia
# ---------------------------------------------------------------------------

@dataclass
class StrategyParams:
    ma_type: str = "EMA"           # "SMA" ou "EMA"
    fast_period: int = 9
    slow_period: int = 21
    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    initial_capital: float = 10000.0
    position_size_pct: float = 100.0  # % do capital alocado por trade


@dataclass
class Trade:
    entry_time: object
    entry_price: float
    direction: str = "long"  # "long" ou "short"
    exit_time: object = None
    exit_price: float = None
    exit_reason: str = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class SMCParams:
    """Parâmetros da estratégia de reversão por Smart Money Concepts:
    varredura de liquidez -> BOS na direção contrária à varredura -> pullback
    de volta à zona varrida -> CHoCH a favor do movimento original (entrada).
    """
    swing_strength: int = 2       # candles de cada lado p/ confirmar um topo/fundo (fractal)
    max_setup_bars: int = 40      # janela máx. (em candles) entre a varredura e a confirmação de entrada
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 2.0
    initial_capital: float = 10000.0
    position_size_pct: float = 100.0


# ---------------------------------------------------------------------------
# Geração de sinais
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    ma_func = sma if params.ma_type == "SMA" else ema

    df["ma_fast"] = ma_func(df["close"], params.fast_period)
    df["ma_slow"] = ma_func(df["close"], params.slow_period)

    if params.use_rsi_filter:
        df["rsi"] = rsi(df["close"], params.rsi_period)

    return df


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Sinal de entrada: cruzamento da média rápida acima da lenta (+ filtro RSI opcional).

    Este motor clássico só opera comprado — 'entry_short' fica sempre falso,
    e o cruzamento de baixa funciona só como saída da posição comprada.
    """
    df = calculate_indicators(df, params)

    cross_up = (df["ma_fast"] > df["ma_slow"]) & (df["ma_fast"].shift(1) <= df["ma_slow"].shift(1))
    cross_down = (df["ma_fast"] < df["ma_slow"]) & (df["ma_fast"].shift(1) >= df["ma_slow"].shift(1))

    if params.use_rsi_filter:
        rsi_ok = (df["rsi"] > params.rsi_oversold) & (df["rsi"] < params.rsi_overbought)
        cross_up = cross_up & rsi_ok

    df["entry_long"] = cross_up
    df["entry_short"] = False
    df["exit_signal"] = cross_down

    return df


def _current_signal_from_df(signaled: pd.DataFrame, reason: str) -> dict:
    last = signaled.iloc[-1]

    if bool(last.get("entry_long", False)):
        signal_type = "Compra"
    elif bool(last.get("entry_short", False)):
        signal_type = "Venda (short)"
    elif bool(last.get("exit_signal", False)):
        signal_type = "Saída"
    else:
        signal_type = None

    return {
        "signal": signal_type,
        "price": float(last["close"]),
        "timestamp": signaled.index[-1],
        "reason": reason,
    }


def get_current_signal(df: pd.DataFrame, params: StrategyParams) -> dict:
    """Inspeciona o último candle e informa se a estratégia clássica geraria um sinal agora.

    Usado pelo painel de alertas do Telegram: não é um motor de trading ao vivo,
    apenas verifica se, no fechamento do candle mais recente, um cruzamento de
    entrada/saída teria sido acionado.
    """
    signaled = generate_signals(df, params)

    reason = f"Cruzamento {params.ma_type} {params.fast_period}/{params.slow_period}"
    if params.use_rsi_filter:
        reason += " + filtro RSI"

    return _current_signal_from_df(signaled, reason)


# ---------------------------------------------------------------------------
# Backtest (compra e venda a descoberto)
# ---------------------------------------------------------------------------

def _simulate(df: pd.DataFrame, stop_loss_pct: float, take_profit_pct: float,
              initial_capital: float, position_size_pct: float):
    """Loop de simulação compartilhado por todas as famílias de estratégia.

    Espera um DataFrame com as colunas 'entry_long', 'entry_short' e
    'exit_signal' já calculadas. Suporta posições compradas e vendidas —
    o P&L e o Stop Loss/Take Profit são espelhados conforme a direção.
    """
    equity = initial_capital
    equity_curve = []
    trades: list[Trade] = []

    position_open = False
    current_trade: Trade | None = None

    for idx, row in df.iterrows():
        price = row["close"]

        if position_open:
            if current_trade.direction == "long":
                change_pct = (price - current_trade.entry_price) / current_trade.entry_price * 100
            else:
                change_pct = (current_trade.entry_price - price) / current_trade.entry_price * 100

            hit_sl = change_pct <= -stop_loss_pct
            hit_tp = change_pct >= take_profit_pct
            hit_cross_exit = bool(row["exit_signal"])

            if hit_sl or hit_tp or hit_cross_exit:
                exit_price = price
                reason = "Stop Loss" if hit_sl else ("Take Profit" if hit_tp else "Cruzamento")

                if current_trade.direction == "long":
                    pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price
                else:
                    pnl_pct = (current_trade.entry_price - exit_price) / current_trade.entry_price
                pnl = current_trade.size * pnl_pct

                current_trade.exit_time = idx
                current_trade.exit_price = exit_price
                current_trade.exit_reason = reason
                current_trade.pnl = pnl
                current_trade.pnl_pct = pnl_pct * 100

                equity += pnl
                trades.append(current_trade)

                position_open = False
                current_trade = None

        elif bool(row.get("entry_long", False)):
            size = equity * (position_size_pct / 100.0)
            current_trade = Trade(entry_time=idx, entry_price=price, size=size, direction="long")
            position_open = True
        elif bool(row.get("entry_short", False)):
            size = equity * (position_size_pct / 100.0)
            current_trade = Trade(entry_time=idx, entry_price=price, size=size, direction="short")
            position_open = True

        equity_curve.append({"timestamp": idx, "equity": equity})

    # Fecha posição aberta ao final do período pelo último preço disponível
    if position_open and current_trade is not None:
        last_price = df["close"].iloc[-1]
        if current_trade.direction == "long":
            pnl_pct = (last_price - current_trade.entry_price) / current_trade.entry_price
        else:
            pnl_pct = (current_trade.entry_price - last_price) / current_trade.entry_price
        pnl = current_trade.size * pnl_pct

        current_trade.exit_time = df.index[-1]
        current_trade.exit_price = last_price
        current_trade.exit_reason = "Fim do período"
        current_trade.pnl = pnl
        current_trade.pnl_pct = pnl_pct * 100

        equity += pnl
        trades.append(current_trade)
        equity_curve[-1]["equity"] = equity

    equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
    return trades, equity_df


def run_backtest(df: pd.DataFrame, params: StrategyParams):
    """Executa o backtest da estratégia clássica (cruzamento de médias)."""
    signaled = generate_signals(df, params)
    return _simulate(signaled, params.stop_loss_pct, params.take_profit_pct,
                      params.initial_capital, params.position_size_pct)


# ---------------------------------------------------------------------------
# Estratégia SMC (Smart Money Concepts): varredura de liquidez -> BOS
# contrário -> pullback até a zona varrida -> CHoCH a favor -> entrada.
# ---------------------------------------------------------------------------
#
# Conceitos de "estrutura de mercado" são discricionários por natureza — não
# existe uma única definição matemática universalmente aceita de BOS/CHoCH.
# A implementação abaixo é uma leitura objetiva e causal (sem lookahead) da
# sequência descrita pelo usuário, não uma réplica perfeita de toda análise
# discricionária que um trader faria manualmente.

def _swing_flags(df: pd.DataFrame, strength: int) -> tuple:
    """Identifica fractais de topo/fundo: um candle cuja máxima (mínima) é a
    maior (menor) entre `strength` candles antes e depois dele.

    Um fractal no índice j só pode ser confirmado depois de vistos os
    `strength` candles seguintes — exatamente como um trader real só
    reconheceria aquele topo/fundo em tempo real, nunca antes.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)

    for i in range(strength, n - strength):
        h_window = highs[i - strength:i + strength + 1]
        if highs[i] == h_window.max():
            is_high[i] = True
        l_window = lows[i - strength:i + strength + 1]
        if lows[i] == l_window.min():
            is_low[i] = True

    return is_high, is_low


def generate_smc_signals(df: pd.DataFrame, params: SMCParams) -> pd.DataFrame:
    """Gera sinais de entrada a partir da sequência: varredura de liquidez de
    um topo/fundo confirmado -> BOS (rompimento de estrutura) na direção
    contrária à varredura -> pullback de volta à zona varrida -> CHoCH
    (novo rompimento de estrutura) a favor do movimento original, que
    dispara a entrada comprada ou vendida.

    Todo o scan é causal: em cada candle só usamos topos/fundos já
    confirmados até aquele ponto (ver `_swing_flags`), então não há
    vazamento de informação futura.
    """
    df = df.copy()
    n = len(df)
    strength = max(1, params.swing_strength)

    is_swing_high, is_swing_low = _swing_flags(df, strength)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)

    last_high_idx = None
    last_low_idx = None
    pending: dict | None = None

    for t in range(n):
        confirm_idx = t - strength
        if confirm_idx >= 0:
            if is_swing_high[confirm_idx]:
                last_high_idx = confirm_idx
            if is_swing_low[confirm_idx]:
                last_low_idx = confirm_idx

        price_high, price_low, price_close = highs[t], lows[t], closes[t]

        if pending is not None:
            pending["bars_left"] -= 1

            if pending["stage"] == "await_bos":
                if pending["direction"] == "short" and last_low_idx is not None and price_close < lows[last_low_idx]:
                    pending["stage"] = "await_pullback"
                    pending["zone_reached"] = False
                elif pending["direction"] == "long" and last_high_idx is not None and price_close > highs[last_high_idx]:
                    pending["stage"] = "await_pullback"
                    pending["zone_reached"] = False

            elif pending["stage"] == "await_pullback":
                if pending["direction"] == "short":
                    if price_high >= pending["sweep_zone"]:
                        pending["zone_reached"] = True
                    if pending["zone_reached"] and last_low_idx is not None and price_close < lows[last_low_idx]:
                        entry_short[t] = True
                        pending = None
                else:
                    if price_low <= pending["sweep_zone"]:
                        pending["zone_reached"] = True
                    if pending["zone_reached"] and last_high_idx is not None and price_close > highs[last_high_idx]:
                        entry_long[t] = True
                        pending = None

            if pending is not None and pending["bars_left"] <= 0:
                pending = None  # setup expirou sem confirmar a entrada

        if pending is None:
            # Varredura de topo: mecha ultrapassa o último topo confirmado,
            # mas o candle fecha de volta abaixo dele (rejeição / stop hunt)
            if last_high_idx is not None and price_high > highs[last_high_idx] and price_close < highs[last_high_idx]:
                pending = {
                    "direction": "short", "stage": "await_bos",
                    "sweep_zone": highs[last_high_idx],
                    "bars_left": params.max_setup_bars,
                }
            elif last_low_idx is not None and price_low < lows[last_low_idx] and price_close > lows[last_low_idx]:
                pending = {
                    "direction": "long", "stage": "await_bos",
                    "sweep_zone": lows[last_low_idx],
                    "bars_left": params.max_setup_bars,
                }

    df["entry_long"] = entry_long
    df["entry_short"] = entry_short
    df["exit_signal"] = False  # saída só por Stop Loss / Take Profit
    return df


def run_backtest_smc(df: pd.DataFrame, params: SMCParams):
    """Executa o backtest da estratégia SMC (varredura + BOS + pullback + CHoCH)."""
    signaled = generate_smc_signals(df, params)
    return _simulate(signaled, params.stop_loss_pct, params.take_profit_pct,
                      params.initial_capital, params.position_size_pct)


def get_current_signal_smc(df: pd.DataFrame, params: SMCParams) -> dict:
    """Mesma ideia de `get_current_signal`, mas para a estratégia SMC."""
    signaled = generate_smc_signals(df, params)
    reason = f"SMC: varredura + BOS + pullback + CHoCH (swing={params.swing_strength})"
    return _current_signal_from_df(signaled, reason)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def calculate_metrics(trades: list[Trade], equity_df: pd.DataFrame, initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "payoff": 0.0, "expectancy": 0.0,
            "net_profit": 0.0, "net_profit_pct": 0.0, "final_equity": initial_capital,
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_trades = len(trades)
    win_rate = (len(wins) / total_trades) * 100 if total_trades else 0.0

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

    loss_rate = 1 - (win_rate / 100)
    expectancy = ((win_rate / 100) * avg_win) - (loss_rate * avg_loss)

    # Drawdown máximo sobre a curva de patrimônio
    equity_series = equity_df["equity"]
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown_pct = abs(drawdown.min()) if len(drawdown) else 0.0

    final_equity = equity_series.iloc[-1] if len(equity_series) else initial_capital
    net_profit = final_equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "payoff": round(payoff, 2) if payoff != float("inf") else 999.99,
        "expectancy": round(expectancy, 2),
        "net_profit": round(net_profit, 2),
        "net_profit_pct": round(net_profit_pct, 2),
        "final_equity": round(final_equity, 2),
    }


# ---------------------------------------------------------------------------
# Modo de otimização (grid search)
# ---------------------------------------------------------------------------

def run_optimization(df: pd.DataFrame, base_params: StrategyParams,
                      fast_range: range, slow_range: range,
                      sl_range: list[float], tp_range: list[float],
                      max_combinations: int = 200) -> pd.DataFrame:
    """Roda múltiplas combinações de parâmetros e retorna um ranking por expectativa."""
    results = []
    combos = list(itertools.product(fast_range, slow_range, sl_range, tp_range))[:max_combinations]

    for fast, slow, sl, tp in combos:
        if fast >= slow:
            continue

        params = StrategyParams(
            ma_type=base_params.ma_type,
            fast_period=fast,
            slow_period=slow,
            use_rsi_filter=base_params.use_rsi_filter,
            rsi_period=base_params.rsi_period,
            rsi_oversold=base_params.rsi_oversold,
            rsi_overbought=base_params.rsi_overbought,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            initial_capital=base_params.initial_capital,
            position_size_pct=base_params.position_size_pct,
        )

        trades, equity_df = run_backtest(df, params)
        metrics = calculate_metrics(trades, equity_df, params.initial_capital)

        results.append({
            "fast_period": fast, "slow_period": slow,
            "stop_loss_pct": sl, "take_profit_pct": tp,
            **metrics,
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("expectancy", ascending=False).reset_index(drop=True)

    return results_df
