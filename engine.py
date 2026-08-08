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
from dataclasses import dataclass, field, replace

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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Retorna (linha MACD, linha de sinal, histograma)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0):
    """Retorna (banda superior, banda média, banda inferior)."""
    middle = sma(series, period)
    rolling_std = series.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (suavização de Wilder)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — força da tendência (0-100), independente da direção."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr_smoothed = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_smoothed.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_smoothed.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0)


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Retorna (%K, %D) do Oscilador Estocástico."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    percent_k = (100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)).fillna(50)
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (-100 a 0)."""
    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()
    wr = (-100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)).fillna(-50)
    return wr


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))).fillna(0)


def donchian_channel(df: pd.DataFrame, period: int = 20):
    """Retorna (canal superior, canal inferior) — máxima/mínima dos últimos `period` candles."""
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    return upper, lower


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    return (prev_close < prev_open) & (df["close"] > df["open"]) & (df["close"] >= prev_open) & (df["open"] <= prev_close)


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    return (prev_close > prev_open) & (df["close"] < df["open"]) & (df["close"] <= prev_open) & (df["open"] >= prev_close)


def is_hammer(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    return (lower_wick >= 2 * body) & (upper_wick <= body) & (body > 0)


def is_shooting_star(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    return (upper_wick >= 2 * body) & (lower_wick <= body) & (body > 0)


# ---------------------------------------------------------------------------
# Configuração da estratégia
# ---------------------------------------------------------------------------

@dataclass
class RiskCostParams:
    """Campos compartilhados por todas as famílias de estratégia: custos
    operacionais (spread/slippage/comissão) e gerenciamento de Stop
    Loss/Take Profit (percentual, pips ou múltiplo de ATR + trailing stop).

    Custos são normalmente uma configuração da CORRETORA/conta, não da
    estratégia em si — por isso ficam expostos como campos fixos na barra
    lateral do app, em vez de serem extraídos pela IA do texto da estratégia.
    """
    # --- Custos operacionais (aplicados na abertura E no fechamento de cada trade) ---
    spread_pct: float = 0.0        # spread do book, em % do preço
    slippage_pct: float = 0.0      # slippage de execução, em % do preço
    commission_pct: float = 0.0    # taxa de corretagem percentual (ex: 0.075 p/ Binance)
    commission_fixed: float = 0.0  # custo fixo em unidades de preço (ex: pips de forex)

    # --- Gerenciamento de Stop Loss / Take Profit ---
    sl_tp_mode: str = "percent"    # "percent" | "pips" | "atr"
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    stop_loss_pips: float = 20.0
    take_profit_pips: float = 40.0
    pip_size: float = 0.0001       # tamanho de 1 pip/ponto no preço do ativo
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0

    # --- Trailing Stop ---
    use_trailing_stop: bool = False
    trailing_activation_pct: float = 1.0  # lucro % necessário pra começar a mover o stop
    trailing_distance_pct: float = 1.0    # distância % mantida em relação ao preço mais favorável

    initial_capital: float = 10000.0
    position_size_pct: float = 100.0  # % do capital alocado por trade


@dataclass
class StrategyParams(RiskCostParams):
    # --- Gatilho de entrada (escolha um) ---
    entry_trigger: str = "ma_cross"  # "ma_cross" | "bb_reversal" | "donchian_breakout" | "candle_pattern"
    ma_type: str = "EMA"             # "SMA" ou "EMA" — usado quando entry_trigger == "ma_cross"
    fast_period: int = 9
    slow_period: int = 21
    bb_period: int = 20              # usado quando entry_trigger == "bb_reversal"
    bb_std: float = 2.0
    donchian_period: int = 20        # usado quando entry_trigger == "donchian_breakout"
    candle_pattern: str = "bullish_engulfing"  # "bullish_engulfing" | "hammer" — usado quando entry_trigger == "candle_pattern"

    # --- Filtros opcionais (combinados com E lógico sobre o gatilho) ---
    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    use_macd_filter: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    use_volume_filter: bool = False
    volume_period: int = 20
    volume_multiplier: float = 1.5

    use_stochastic_filter: bool = False
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: int = 20
    stoch_overbought: int = 80

    use_adx_filter: bool = False
    adx_period: int = 14
    adx_threshold: float = 25.0

    use_williams_filter: bool = False
    williams_period: int = 14
    williams_oversold: float = -80.0
    williams_overbought: float = -20.0

    use_cci_filter: bool = False
    cci_period: int = 20
    cci_oversold: float = -100.0
    cci_overbought: float = 100.0

    use_triple_ma: bool = False   # exige alinhamento total: ma_fast > ma_mid > ma_slow
    mid_period: int = 14


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
class SMCParams(RiskCostParams):
    """Parâmetros da estratégia de reversão por Smart Money Concepts:
    varredura de liquidez -> BOS na direção contrária à varredura -> pullback
    de volta à zona varrida -> CHoCH a favor do movimento original (entrada).
    """
    swing_strength: int = 2       # candles de cada lado p/ confirmar um topo/fundo (fractal)
    max_setup_bars: int = 40      # janela máx. (em candles) entre a varredura e a confirmação de entrada


# ---------------------------------------------------------------------------
# Geração de sinais
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    ma_func = sma if params.ma_type == "SMA" else ema

    df["ma_fast"] = ma_func(df["close"], params.fast_period)
    df["ma_slow"] = ma_func(df["close"], params.slow_period)

    if params.use_triple_ma:
        df["ma_mid"] = ma_func(df["close"], params.mid_period)

    if params.entry_trigger == "bb_reversal":
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(df["close"], params.bb_period, params.bb_std)
    if params.entry_trigger == "donchian_breakout":
        df["donchian_upper"], df["donchian_lower"] = donchian_channel(df, params.donchian_period)

    if params.use_rsi_filter:
        df["rsi"] = rsi(df["close"], params.rsi_period)
    if params.use_macd_filter:
        df["macd_line"], df["macd_signal"], _ = macd(df["close"], params.macd_fast, params.macd_slow, params.macd_signal)
    if params.use_volume_filter:
        df["volume_ma"] = sma(df["volume"], params.volume_period)
    if params.use_stochastic_filter:
        df["stoch_k"], df["stoch_d"] = stochastic(df, params.stoch_k_period, params.stoch_d_period)
    if params.use_adx_filter:
        df["adx"] = adx(df, params.adx_period)
    if params.use_williams_filter:
        df["williams_r"] = williams_r(df, params.williams_period)
    if params.use_cci_filter:
        df["cci"] = cci(df, params.cci_period)

    return df


def required_warmup_bars(params: StrategyParams) -> int:
    """Nº mínimo de candles necessários para todos os indicadores ativados terem dados válidos."""
    periods = [params.slow_period]
    if params.entry_trigger == "bb_reversal":
        periods.append(params.bb_period)
    if params.entry_trigger == "donchian_breakout":
        periods.append(params.donchian_period)
    if params.use_rsi_filter:
        periods.append(params.rsi_period)
    if params.use_macd_filter:
        periods.append(params.macd_slow + params.macd_signal)
    if params.use_volume_filter:
        periods.append(params.volume_period)
    if params.use_stochastic_filter:
        periods.append(params.stoch_k_period + params.stoch_d_period)
    if params.use_adx_filter:
        periods.append(params.adx_period * 2)
    if params.use_williams_filter:
        periods.append(params.williams_period)
    if params.use_cci_filter:
        periods.append(params.cci_period)
    if params.use_triple_ma:
        periods.append(params.mid_period)
    if params.sl_tp_mode == "atr":
        periods.append(params.atr_period)
    return max(periods) + 10


def required_warmup_bars_smc(params: SMCParams) -> int:
    """Nº mínimo de candles necessários para a estratégia SMC (estrutura + ATR se aplicável)."""
    periods = [params.swing_strength * 2]
    if params.sl_tp_mode == "atr":
        periods.append(params.atr_period)
    return max(periods) + 10


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Sinal de entrada a partir do gatilho escolhido (`entry_trigger`), com os
    filtros opcionais ativados combinados por E lógico.

    Este motor clássico só opera comprado — 'entry_short' fica sempre falso.
    O sinal contrário ao gatilho funciona só como saída da posição comprada.
    """
    df = calculate_indicators(df, params)

    if params.entry_trigger == "bb_reversal":
        cross_up = (df["close"] > df["bb_lower"]) & (df["close"].shift(1) <= df["bb_lower"].shift(1))
        cross_down = (df["close"] < df["bb_upper"]) & (df["close"].shift(1) >= df["bb_upper"].shift(1))
    elif params.entry_trigger == "donchian_breakout":
        cross_up = df["close"] > df["donchian_upper"].shift(1)
        cross_down = df["close"] < df["donchian_lower"].shift(1)
    elif params.entry_trigger == "candle_pattern":
        if params.candle_pattern == "hammer":
            cross_up = is_hammer(df)
            cross_down = is_shooting_star(df)
        else:
            cross_up = is_bullish_engulfing(df)
            cross_down = is_bearish_engulfing(df)
    else:  # "ma_cross" (padrão)
        cross_up = (df["ma_fast"] > df["ma_slow"]) & (df["ma_fast"].shift(1) <= df["ma_slow"].shift(1))
        cross_down = (df["ma_fast"] < df["ma_slow"]) & (df["ma_fast"].shift(1) >= df["ma_slow"].shift(1))

    if params.use_rsi_filter:
        cross_up = cross_up & (df["rsi"] > params.rsi_oversold) & (df["rsi"] < params.rsi_overbought)
    if params.use_macd_filter:
        cross_up = cross_up & (df["macd_line"] > df["macd_signal"])
    if params.use_volume_filter:
        cross_up = cross_up & (df["volume"] > df["volume_ma"] * params.volume_multiplier)
    if params.use_stochastic_filter:
        cross_up = cross_up & (df["stoch_k"] > params.stoch_oversold) & (df["stoch_k"] < params.stoch_overbought)
    if params.use_adx_filter:
        cross_up = cross_up & (df["adx"] > params.adx_threshold)
    if params.use_williams_filter:
        cross_up = cross_up & (df["williams_r"] > params.williams_oversold) & (df["williams_r"] < params.williams_overbought)
    if params.use_cci_filter:
        cross_up = cross_up & (df["cci"] > params.cci_oversold) & (df["cci"] < params.cci_overbought)
    if params.use_triple_ma:
        cross_up = cross_up & (df["ma_fast"] > df["ma_mid"]) & (df["ma_mid"] > df["ma_slow"])
        cross_down = cross_down & (df["ma_fast"] < df["ma_mid"]) & (df["ma_mid"] < df["ma_slow"])

    df["entry_long"] = cross_up.fillna(False)
    df["entry_short"] = False
    df["exit_signal"] = cross_down.fillna(False)

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


def describe_trigger(params: StrategyParams) -> str:
    if params.entry_trigger == "bb_reversal":
        return f"Reversão Bollinger ({params.bb_period}, {params.bb_std})"
    if params.entry_trigger == "donchian_breakout":
        return f"Rompimento Donchian ({params.donchian_period})"
    if params.entry_trigger == "candle_pattern":
        label = "Engolfo de Alta" if params.candle_pattern == "bullish_engulfing" else "Martelo"
        return f"Padrão de Candle: {label}"
    return f"Cruzamento {params.ma_type} {params.fast_period}/{params.slow_period}"


def describe_active_filters(params: StrategyParams) -> list:
    labels = []
    if params.use_rsi_filter:
        labels.append("RSI")
    if params.use_macd_filter:
        labels.append("MACD")
    if params.use_volume_filter:
        labels.append("Volume")
    if params.use_stochastic_filter:
        labels.append("Estocástico")
    if params.use_adx_filter:
        labels.append("ADX")
    if params.use_williams_filter:
        labels.append("Williams %R")
    if params.use_cci_filter:
        labels.append("CCI")
    if params.use_triple_ma:
        labels.append(f"Alinhamento 3 Médias (+{params.mid_period})")
    return labels


def describe_risk(params) -> str:
    """Descreve o gerenciamento de Stop Loss/Take Profit (e trailing, se ativo)."""
    if params.sl_tp_mode == "pips":
        base = f"SL {params.stop_loss_pips}pips / TP {params.take_profit_pips}pips"
    elif params.sl_tp_mode == "atr":
        base = f"SL {params.atr_sl_multiplier}x ATR({params.atr_period}) / TP {params.atr_tp_multiplier}x ATR({params.atr_period})"
    else:
        base = f"SL {params.stop_loss_pct}% / TP {params.take_profit_pct}%"
    if params.use_trailing_stop:
        base += f" · trailing após {params.trailing_activation_pct}% (dist. {params.trailing_distance_pct}%)"
    return base


def get_current_signal(df: pd.DataFrame, params: StrategyParams) -> dict:
    """Inspeciona o último candle e informa se a estratégia clássica geraria um sinal agora.

    Usado pelo painel de alertas do Telegram: não é um motor de trading ao vivo,
    apenas verifica se, no fechamento do candle mais recente, um cruzamento de
    entrada/saída teria sido acionado.
    """
    signaled = generate_signals(df, params)

    reason = describe_trigger(params)
    filters = describe_active_filters(params)
    if filters:
        reason += " + filtro " + "/".join(filters)

    return _current_signal_from_df(signaled, reason)


# ---------------------------------------------------------------------------
# Backtest (compra e venda a descoberto)
# ---------------------------------------------------------------------------

def _apply_costs(price: float, direction: str, side: str, params) -> float:
    """Aplica spread + slippage + comissão como uma piora no preço de execução
    (sempre desfavorável ao trader), tanto na entrada quanto na saída do trade.
    """
    pct_cost = (params.spread_pct + params.slippage_pct + params.commission_pct) / 100.0
    fixed_cost_pct = (params.commission_fixed / price) if price else 0.0
    total_pct = pct_cost + fixed_cost_pct

    worse_up = (direction == "long" and side == "entry") or (direction == "short" and side == "exit")
    return price * (1 + total_pct) if worse_up else price * (1 - total_pct)


def _resolve_sl_tp_pct(params, entry_price: float, atr_value) -> tuple:
    """Resolve o Stop Loss/Take Profit (em % do preço de entrada) conforme o
    modo escolhido: percentual direto, pips/pontos fixos, ou múltiplo de ATR.
    """
    if params.sl_tp_mode == "pips":
        sl_pct = (params.stop_loss_pips * params.pip_size / entry_price) * 100
        tp_pct = (params.take_profit_pips * params.pip_size / entry_price) * 100
    elif params.sl_tp_mode == "atr" and atr_value and not np.isnan(atr_value):
        sl_pct = (params.atr_sl_multiplier * atr_value / entry_price) * 100
        tp_pct = (params.atr_tp_multiplier * atr_value / entry_price) * 100
    else:
        sl_pct, tp_pct = params.stop_loss_pct, params.take_profit_pct

    return max(sl_pct, 0.01), max(tp_pct, 0.01)


def _simulate(df: pd.DataFrame, params):
    """Loop de simulação compartilhado por todas as famílias de estratégia.

    Espera um DataFrame com as colunas 'entry_long', 'entry_short' e
    'exit_signal' já calculadas (mais 'open'/'high'/'low'/'close'). Suporta
    posições compradas e vendidas, custos operacionais (spread/slippage/
    comissão), Stop Loss/Take Profit em % / pips / múltiplo de ATR, e
    trailing stop.

    Indicadores usados pelo motor (ATR, quando `sl_tp_mode == "atr"`) são
    pré-calculados de forma vetorizada em pandas antes do loop — o loop em si
    só faz leituras O(1) num array numpy, pois o estado de uma posição aberta
    (trailing stop, P&L acumulado) é inerentemente sequencial e não dá pra
    vetorizar sem reescrever o motor como uma simulação orientada a eventos.
    """
    if params.sl_tp_mode == "atr":
        atr_arr = atr(df, params.atr_period).to_numpy()
    else:
        atr_arr = None

    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    entry_long_arr = df["entry_long"].to_numpy()
    entry_short_arr = df["entry_short"].to_numpy()
    exit_signal_arr = df["exit_signal"].to_numpy()
    idx_arr = df.index

    equity = params.initial_capital
    equity_curve = []
    trades: list[Trade] = []

    position_open = False
    current_trade: Trade | None = None
    stop_price = take_price = best_price = None
    trailing_engaged = False

    for t in range(len(df)):
        idx = idx_arr[t]
        high, low, close = high_arr[t], low_arr[t], close_arr[t]

        if position_open:
            direction = current_trade.direction

            if direction == "long":
                best_price = max(best_price, high)
                if params.use_trailing_stop:
                    profit_pct = (best_price - current_trade.entry_price) / current_trade.entry_price * 100
                    if profit_pct >= params.trailing_activation_pct:
                        candidate = best_price * (1 - params.trailing_distance_pct / 100)
                        if candidate > stop_price:
                            stop_price = candidate
                            trailing_engaged = True
                hit_sl = low <= stop_price
                hit_tp = high >= take_price
            else:
                best_price = min(best_price, low)
                if params.use_trailing_stop:
                    profit_pct = (current_trade.entry_price - best_price) / current_trade.entry_price * 100
                    if profit_pct >= params.trailing_activation_pct:
                        candidate = best_price * (1 + params.trailing_distance_pct / 100)
                        if candidate < stop_price:
                            stop_price = candidate
                            trailing_engaged = True
                hit_sl = high >= stop_price
                hit_tp = low <= take_price

            hit_cross_exit = bool(exit_signal_arr[t])

            if hit_sl or hit_tp or hit_cross_exit:
                if hit_sl:
                    raw_exit_price = stop_price
                    reason = "Trailing Stop" if trailing_engaged else "Stop Loss"
                elif hit_tp:
                    raw_exit_price, reason = take_price, "Take Profit"
                else:
                    raw_exit_price, reason = close, "Cruzamento"

                exit_price = _apply_costs(raw_exit_price, direction, "exit", params)

                if direction == "long":
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
                stop_price = take_price = best_price = None
                trailing_engaged = False

        elif bool(entry_long_arr[t]) or bool(entry_short_arr[t]):
            direction = "long" if entry_long_arr[t] else "short"
            entry_price = _apply_costs(close, direction, "entry", params)

            atr_value = atr_arr[t] if atr_arr is not None else None
            sl_pct, tp_pct = _resolve_sl_tp_pct(params, entry_price, atr_value)

            if direction == "long":
                stop_price = entry_price * (1 - sl_pct / 100)
                take_price = entry_price * (1 + tp_pct / 100)
            else:
                stop_price = entry_price * (1 + sl_pct / 100)
                take_price = entry_price * (1 - tp_pct / 100)
            best_price = entry_price

            size = equity * (params.position_size_pct / 100.0)
            current_trade = Trade(entry_time=idx, entry_price=entry_price, size=size, direction=direction)
            position_open = True

        equity_curve.append({"timestamp": idx, "equity": equity})

    # Fecha posição aberta ao final do período pelo último preço disponível
    if position_open and current_trade is not None:
        exit_price = _apply_costs(close_arr[-1], current_trade.direction, "exit", params)
        if current_trade.direction == "long":
            pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price
        else:
            pnl_pct = (current_trade.entry_price - exit_price) / current_trade.entry_price
        pnl = current_trade.size * pnl_pct

        current_trade.exit_time = df.index[-1]
        current_trade.exit_price = exit_price
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
    return _simulate(signaled, params)


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
    return _simulate(signaled, params)


def get_current_signal_smc(df: pd.DataFrame, params: SMCParams) -> dict:
    """Mesma ideia de `get_current_signal`, mas para a estratégia SMC."""
    signaled = generate_smc_signals(df, params)
    reason = f"SMC: varredura + BOS + pullback + CHoCH (swing={params.swing_strength})"
    return _current_signal_from_df(signaled, reason)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _infer_periods_per_year(equity_df: pd.DataFrame) -> float:
    """Estima quantos candles cabem em um ano, a partir do intervalo médio
    entre candles do equity_df — usado para anualizar o Sharpe Ratio."""
    if len(equity_df) < 2:
        return 252.0
    avg_delta_seconds = (equity_df.index[-1] - equity_df.index[0]).total_seconds() / (len(equity_df) - 1)
    if avg_delta_seconds <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / avg_delta_seconds


def _sharpe_ratio(equity_df: pd.DataFrame) -> float:
    """Sharpe Ratio anualizado sobre os retornos por candle do equity_df
    (taxa livre de risco assumida em 0, prática comum em backtests de varejo)."""
    equity_series = equity_df["equity"]
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    periods_per_year = _infer_periods_per_year(equity_df)
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _max_consecutive_losses(trades: list[Trade]) -> int:
    """Maior sequência de trades perdedores seguidos (vetorizado via groupby)."""
    if not trades:
        return 0
    is_loss = pd.Series([t.pnl <= 0 for t in trades])
    groups = (is_loss != is_loss.shift()).cumsum()
    streaks = is_loss.groupby(groups).transform("size") * is_loss
    return int(streaks.max()) if len(streaks) else 0


def _avg_trade_duration(trades: list[Trade]) -> pd.Timedelta:
    durations = [t.exit_time - t.entry_time for t in trades if t.exit_time is not None]
    if not durations:
        return pd.Timedelta(0)
    return pd.Series(durations).mean()


def _max_drawdown_duration(equity_df: pd.DataFrame) -> pd.Timedelta:
    """Maior tempo decorrido entre um novo pico de patrimônio e o momento em
    que o patrimônio volta a alcançá-lo (ou o fim dos dados, se nunca recuperar).
    Totalmente vetorizado: sem loop explícito sobre os candles.
    """
    equity_series = equity_df["equity"]
    if len(equity_series) < 2:
        return pd.Timedelta(0)

    running_max = equity_series.cummax()
    at_peak = equity_series >= running_max

    peak_times = equity_series.index.to_series().where(at_peak).ffill()
    duration = equity_series.index.to_series() - peak_times

    max_duration = duration.max()
    return max_duration if pd.notna(max_duration) else pd.Timedelta(0)


def format_timedelta(td: pd.Timedelta) -> str:
    """Formata um Timedelta de forma compacta e legível (ex: '2d 5h', '45min')."""
    if td is None or pd.isna(td) or td <= pd.Timedelta(0):
        return "—"

    total_seconds = int(td.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60

    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}min"
    if minutes > 0:
        return f"{minutes}min"
    return f"{total_seconds}s"


def calculate_metrics(trades: list[Trade], equity_df: pd.DataFrame, initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "payoff": 0.0, "expectancy": 0.0,
            "net_profit": 0.0, "net_profit_pct": 0.0, "final_equity": initial_capital,
            "sharpe_ratio": 0.0, "max_consecutive_losses": 0,
            "avg_trade_duration": pd.Timedelta(0), "max_drawdown_duration": pd.Timedelta(0),
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
        "sharpe_ratio": round(_sharpe_ratio(equity_df), 2),
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "avg_trade_duration": _avg_trade_duration(trades),
        "max_drawdown_duration": _max_drawdown_duration(equity_df),
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

        # Preserva todos os outros campos (filtros, tipo de média etc.) do
        # base_params — só o grid varia período rápido/lento e SL/TP.
        params = replace(base_params, fast_period=fast, slow_period=slow,
                          stop_loss_pct=sl, take_profit_pct=tp)

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
