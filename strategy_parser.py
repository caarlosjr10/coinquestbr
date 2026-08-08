"""
strategy_parser.py — Interpreta, via IA, a descrição em texto livre da
estratégia do cliente e converte em parâmetros estruturados para o motor de
backtest (engine.StrategyParams).

Função separada do parecer técnico final: usa sempre `claude-haiku-4-5`
(o modelo mais barato) e NÃO consome o limite mensal de "Análises de IA" do
plano do cliente — é uma etapa de configuração, não uma entrega de análise.
"""

import json
import os

import anthropic
import streamlit as st

PARSER_MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 1500

SYSTEM_PROMPT = (
    "Você é um interpretador de estratégias de trading. Leia a descrição em linguagem "
    "natural escrita pelo usuário e decida qual das DUAS famílias de motor de backtest "
    "melhor representa a estratégia dele — o motor NÃO executa código arbitrário, ele só "
    "sabe rodar essas duas famílias (com suas peças configuráveis), então sua tarefa é "
    "encaixar a estratégia numa delas da forma mais fiel possível.\n"
    "\n"
    "FAMÍLIA 'classic' — um GATILHO de entrada (escolha um) + FILTROS opcionais "
    "(combine quantos fizerem sentido) + Stop Loss % / Take Profit %.\n"
    "\n"
    "Gatilhos disponíveis ('entry_trigger'):\n"
    "- 'ma_cross' (padrão): cruzamento de duas médias móveis (SMA/EMA) — 'fast_period' "
    "cruza para cima de 'slow_period'. Use quando o usuário falar de médias móveis, "
    "cruzamento, 'EMA9 e EMA21', tendência simples, ou quando nenhum outro gatilho se "
    "aplicar mas a ideia geral (indicadores + SL/TP) ainda for representável.\n"
    "- 'bb_reversal': o preço fecha fora da Banda de Bollinger e depois fecha de volta "
    "para dentro dela (reversão à média). Use para 'toca a banda e reverte', 'sobrevenda "
    "na Bollinger', 'reversão à média'. Parâmetros: 'bb_period', 'bb_std'.\n"
    "- 'donchian_breakout': o preço rompe a máxima (ou mínima) dos últimos N candles. Use "
    "para 'rompimento de máxima de N períodos', 'breakout do canal', 'nova máxima de N "
    "candles'. Parâmetro: 'donchian_period' = N.\n"
    "- 'candle_pattern': padrão de candle isolado. Use para 'engolfo de alta/baixa' "
    "('candle_pattern'='bullish_engulfing') ou 'martelo'/'hammer'/'estrela cadente' "
    "('candle_pattern'='hammer').\n"
    "\n"
    "Filtros opcionais (ative 'use_X_filter'=true só se o usuário mencionar aquele "
    "indicador; preencha os parâmetros dele com valores padrão razoáveis se ele não "
    "detalhar): RSI ('use_rsi_filter', períodos/sobrevenda/sobrecompra), MACD "
    "('use_macd_filter', períodos rápido/lento/sinal — exige linha MACD acima da linha "
    "de sinal), Volume ('use_volume_filter', 'volume_period', 'volume_multiplier' — "
    "exige volume acima da média vezes o multiplicador), Estocástico "
    "('use_stochastic_filter', períodos %K/%D, sobrevenda/sobrecompra), ADX "
    "('use_adx_filter', 'adx_period', 'adx_threshold' — exige força de tendência acima "
    "do limiar, para 'só entrar em tendência forte'), Williams %R "
    "('use_williams_filter', período, sobrevenda/sobrecompra, escala -100 a 0), CCI "
    "('use_cci_filter', período, sobrevenda/sobrecompra), Alinhamento de 3 Médias "
    "('use_triple_ma' — exige 'ma_fast' > 'ma_mid' > 'ma_slow' todas na mesma direção "
    "antes de disparar o sinal; use quando o usuário pedir para checar 3 médias móveis "
    "alinhadas ao mesmo tempo). IMPORTANTE: ordene os 3 períodos citados do MENOR para "
    "o MAIOR — o menor vira 'fast_period', o do meio vira 'mid_period', e o MAIOR (não "
    "necessariamente o segundo mencionado no texto) vira 'slow_period'. Ex: 'média de 9, "
    "21 e 50 alinhadas' → fast_period=9, mid_period=21, slow_period=50.\n"
    "\n"
    "Gerenciamento de Stop Loss / Take Profit ('sl_tp_mode', escolha um — vale para "
    "QUALQUER família, 'classic' ou 'smc'):\n"
    "- 'percent' (padrão): 'stop_loss_pct'/'take_profit_pct' como % do preço de entrada.\n"
    "- 'pips': stop/alvo como distância fixa em pips/pontos — preencha "
    "'stop_loss_pips'/'take_profit_pips'. Use quando o usuário falar de pips, pontos "
    "fixos, ou distância fixa de preço (comum em Forex).\n"
    "- 'atr': stop/alvo como múltiplo do ATR (Average True Range, mede volatilidade) — "
    "preencha 'atr_period', 'atr_sl_multiplier', 'atr_tp_multiplier' (ex: 'stop em 1.5x "
    "ATR e alvo em 3x ATR' → atr_sl_multiplier=1.5, atr_tp_multiplier=3.0).\n"
    "\n"
    "Trailing Stop: se o usuário mencionar 'stop móvel', 'trailing stop', 'proteger "
    "lucro conforme o preço anda a favor', ative 'use_trailing_stop'=true e preencha "
    "'trailing_activation_pct' (lucro % necessário pra começar a mover o stop) e "
    "'trailing_distance_pct' (distância % que o stop mantém do preço mais favorável).\n"
    "\n"
    "FAMÍLIA 'smc' — reversão por Smart Money Concepts, com esta sequência fixa: "
    "(1) varredura de liquidez (o preço rompe um topo ou fundo recente só com o pavio e "
    "fecha de volta, tipo um 'stop hunt'), (2) BOS (rompimento de estrutura) na direção "
    "CONTRÁRIA à varredura, (3) pullback de volta até a zona varrida, (4) CHoCH (novo "
    "rompimento de estrutura) a favor do movimento original, que dispara a entrada. Use "
    "esta família quando o usuário mencionar conceitos como: varredura/captura de "
    "liquidez, stop hunt, BOS, CHoCH, quebra de estrutura, order block, fair value gap, "
    "pullback pós-rompimento, ou qualquer setup de reversão baseado em topos/fundos do "
    "próprio preço (price action / estrutura de mercado) em vez de indicadores.\n"
    "\n"
    "Se a essência da estratégia puder ser mapeada para uma dessas duas famílias (mesmo "
    "que de forma aproximada — ex: o usuário não especificou o stop, então você completa "
    "com um valor padrão razoável), defina 'is_supported' como true, 'strategy_family' "
    "com a família escolhida, e preencha os parâmetros correspondentes. Para a família "
    "'smc', se o usuário mencionar uma proporção de risco:retorno (ex: 'RR 1:2', 'risco "
    "retorno 1:3'), converta para 'stop_loss_pct' (use 1.0 como padrão razoável se não "
    "houver stop explícito) e 'take_profit_pct' = stop_loss_pct multiplicado pela "
    "proporção informada.\n"
    "\n"
    "Se a estratégia depender de algo que NENHUMA das duas famílias representa (ex: "
    "dados fundamentalistas/notícias, book de ofertas, múltiplos timeframes combinados "
    "numa mesma regra, ou qualquer coisa vaga demais para virar regras objetivas), "
    "defina 'is_supported' como false. Nesse caso ainda preencha todos os campos com os "
    "valores padrão (serão ignorados pela aplicação).\n"
    "\n"
    "Além dos parâmetros da estratégia, extraia também:\n"
    "- 'symbol': o ativo específico mencionado no texto, já no formato correto para busca "
    "de dados (ex: 'Bitcoin' → 'BTC/USDT', 'Petrobras' → 'PETR4.SA', 'Euro Dólar' ou "
    "'EUR/USD' → 'EURUSD=X', 'Ouro' → 'GC=F', 'S&P 500' → '^GSPC', 'Apple' → 'AAPL', "
    "'Ibovespa' → '^BVSP'). Se nenhum ativo específico for mencionado, retorne uma string "
    "vazia \"\" (o app usa um ativo padrão da categoria de mercado escolhida pelo "
    "usuário).\n"
    "\n\n"
    "REGRAS PARA 'interpretation_notes' (muito importantes):\n"
    "- Escreva no MÁXIMO 2 frases curtas, em português informal e direto.\n"
    "- Fale DIRETAMENTE com o usuário, na segunda pessoa ('você'). NUNCA escreva as "
    "palavras 'o cliente' ou 'o usuário' em terceira pessoa — quem lê o texto é a "
    "própria pessoa que escreveu a estratégia.\n"
    "- Se is_supported=true: confirme rapidamente o que você entendeu (gatilho + "
    "filtros), e só se algo precisou ser aproximado/simplificado, diga isso em poucas "
    "palavras — sem jargão técnico, sem explicar o motivo em detalhe.\n"
    "- Se is_supported=false: diga em 1 frase o que não deu para configurar e sugira "
    "rapidamente, em 1 frase, como reformular a estratégia usando os gatilhos/filtros "
    "disponíveis ou os passos de varredura+BOS+pullback+CHoCH.\n"
    "- Nunca escreva um parágrafo longo. Responda como se fosse uma mensagem de chat, "
    "curta e direta."
)

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_supported": {"type": "boolean"},
        "strategy_family": {"type": "string", "enum": ["classic", "smc"]},

        "entry_trigger": {"type": "string", "enum": ["ma_cross", "bb_reversal", "donchian_breakout", "candle_pattern"]},
        "ma_type": {"type": "string", "enum": ["SMA", "EMA"]},
        "fast_period": {"type": "integer"},
        "slow_period": {"type": "integer"},
        "bb_period": {"type": "integer"},
        "bb_std": {"type": "number"},
        "donchian_period": {"type": "integer"},
        "candle_pattern": {"type": "string", "enum": ["bullish_engulfing", "hammer"]},

        "use_rsi_filter": {"type": "boolean"},
        "rsi_period": {"type": "integer"},
        "rsi_oversold": {"type": "integer"},
        "rsi_overbought": {"type": "integer"},

        "use_macd_filter": {"type": "boolean"},
        "macd_fast": {"type": "integer"},
        "macd_slow": {"type": "integer"},
        "macd_signal": {"type": "integer"},

        "use_volume_filter": {"type": "boolean"},
        "volume_period": {"type": "integer"},
        "volume_multiplier": {"type": "number"},

        "use_stochastic_filter": {"type": "boolean"},
        "stoch_k_period": {"type": "integer"},
        "stoch_d_period": {"type": "integer"},
        "stoch_oversold": {"type": "integer"},
        "stoch_overbought": {"type": "integer"},

        "use_adx_filter": {"type": "boolean"},
        "adx_period": {"type": "integer"},
        "adx_threshold": {"type": "number"},

        "use_williams_filter": {"type": "boolean"},
        "williams_period": {"type": "integer"},
        "williams_oversold": {"type": "number"},
        "williams_overbought": {"type": "number"},

        "use_cci_filter": {"type": "boolean"},
        "cci_period": {"type": "integer"},
        "cci_oversold": {"type": "number"},
        "cci_overbought": {"type": "number"},

        "use_triple_ma": {"type": "boolean"},
        "mid_period": {"type": "integer"},

        "swing_strength": {"type": "integer"},
        "max_setup_bars": {"type": "integer"},

        "sl_tp_mode": {"type": "string", "enum": ["percent", "pips", "atr"]},
        "stop_loss_pct": {"type": "number"},
        "take_profit_pct": {"type": "number"},
        "stop_loss_pips": {"type": "number"},
        "take_profit_pips": {"type": "number"},
        "atr_period": {"type": "integer"},
        "atr_sl_multiplier": {"type": "number"},
        "atr_tp_multiplier": {"type": "number"},

        "use_trailing_stop": {"type": "boolean"},
        "trailing_activation_pct": {"type": "number"},
        "trailing_distance_pct": {"type": "number"},

        "symbol": {"type": "string"},
        "interpretation_notes": {"type": "string"},
    },
    "required": [
        "is_supported", "strategy_family",
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
        "swing_strength", "max_setup_bars",
        "sl_tp_mode", "stop_loss_pct", "take_profit_pct", "stop_loss_pips", "take_profit_pips",
        "atr_period", "atr_sl_multiplier", "atr_tp_multiplier",
        "use_trailing_stop", "trailing_activation_pct", "trailing_distance_pct",
        "symbol", "interpretation_notes",
    ],
    "additionalProperties": False,
}

_DEFAULTS = {
    "is_supported": False,
    "strategy_family": "classic",

    "entry_trigger": "ma_cross",
    "ma_type": "EMA",
    "fast_period": 9,
    "slow_period": 21,
    "bb_period": 20,
    "bb_std": 2.0,
    "donchian_period": 20,
    "candle_pattern": "bullish_engulfing",

    "use_rsi_filter": False,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,

    "use_macd_filter": False,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    "use_volume_filter": False,
    "volume_period": 20,
    "volume_multiplier": 1.5,

    "use_stochastic_filter": False,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_oversold": 20,
    "stoch_overbought": 80,

    "use_adx_filter": False,
    "adx_period": 14,
    "adx_threshold": 25.0,

    "use_williams_filter": False,
    "williams_period": 14,
    "williams_oversold": -80.0,
    "williams_overbought": -20.0,

    "use_cci_filter": False,
    "cci_period": 20,
    "cci_oversold": -100.0,
    "cci_overbought": 100.0,

    "use_triple_ma": False,
    "mid_period": 14,

    "swing_strength": 2,
    "max_setup_bars": 40,

    "sl_tp_mode": "percent",
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "stop_loss_pips": 20.0,
    "take_profit_pips": 40.0,
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,
    "atr_tp_multiplier": 3.0,

    "use_trailing_stop": False,
    "trailing_activation_pct": 1.0,
    "trailing_distance_pct": 1.0,

    "symbol": "",
    "interpretation_notes": "",
}


def _clamp(parsed: dict) -> dict:
    """Aplica limites de segurança (o JSON Schema da API não suporta min/max)."""
    result = {**_DEFAULTS, **parsed}

    if result["strategy_family"] not in ("classic", "smc"):
        result["strategy_family"] = "classic"
    if result["entry_trigger"] not in ("ma_cross", "bb_reversal", "donchian_breakout", "candle_pattern"):
        result["entry_trigger"] = "ma_cross"
    if result["candle_pattern"] not in ("bullish_engulfing", "hammer"):
        result["candle_pattern"] = "bullish_engulfing"
    if result["ma_type"] not in ("SMA", "EMA"):
        result["ma_type"] = "EMA"

    result["fast_period"] = max(2, min(200, int(result["fast_period"])))
    result["slow_period"] = max(3, min(400, int(result["slow_period"])))
    if result["fast_period"] >= result["slow_period"]:
        result["slow_period"] = result["fast_period"] + 5

    result["bb_period"] = max(5, min(200, int(result["bb_period"])))
    result["bb_std"] = max(0.5, min(5.0, float(result["bb_std"])))
    result["donchian_period"] = max(5, min(200, int(result["donchian_period"])))

    result["rsi_period"] = max(2, min(100, int(result["rsi_period"])))
    result["rsi_oversold"] = max(1, min(50, int(result["rsi_oversold"])))
    result["rsi_overbought"] = max(50, min(99, int(result["rsi_overbought"])))

    result["macd_fast"] = max(2, min(100, int(result["macd_fast"])))
    result["macd_slow"] = max(3, min(200, int(result["macd_slow"])))
    if result["macd_fast"] >= result["macd_slow"]:
        result["macd_slow"] = result["macd_fast"] + 5
    result["macd_signal"] = max(2, min(100, int(result["macd_signal"])))

    result["volume_period"] = max(2, min(200, int(result["volume_period"])))
    result["volume_multiplier"] = max(1.0, min(10.0, float(result["volume_multiplier"])))

    result["stoch_k_period"] = max(2, min(100, int(result["stoch_k_period"])))
    result["stoch_d_period"] = max(1, min(50, int(result["stoch_d_period"])))
    result["stoch_oversold"] = max(1, min(50, int(result["stoch_oversold"])))
    result["stoch_overbought"] = max(50, min(99, int(result["stoch_overbought"])))

    result["adx_period"] = max(2, min(100, int(result["adx_period"])))
    result["adx_threshold"] = max(5.0, min(60.0, float(result["adx_threshold"])))

    result["williams_period"] = max(2, min(100, int(result["williams_period"])))
    result["williams_oversold"] = max(-100.0, min(-50.0, float(result["williams_oversold"])))
    result["williams_overbought"] = max(-50.0, min(0.0, float(result["williams_overbought"])))

    result["cci_period"] = max(2, min(200, int(result["cci_period"])))
    result["cci_oversold"] = max(-300.0, min(-50.0, float(result["cci_oversold"])))
    result["cci_overbought"] = max(50.0, min(300.0, float(result["cci_overbought"])))

    result["use_triple_ma"] = bool(result["use_triple_ma"])
    result["mid_period"] = max(2, min(300, int(result["mid_period"])))
    if result["use_triple_ma"] and not (result["fast_period"] < result["mid_period"] < result["slow_period"]):
        if result["slow_period"] - result["fast_period"] >= 2:
            result["mid_period"] = max(result["fast_period"] + 1, min(result["slow_period"] - 1, result["mid_period"]))
        else:
            # gap entre fast/slow pequeno demais pra caber uma média intermediária distinta
            result["use_triple_ma"] = False

    result["swing_strength"] = max(1, min(10, int(result["swing_strength"])))
    result["max_setup_bars"] = max(5, min(200, int(result["max_setup_bars"])))

    if result["sl_tp_mode"] not in ("percent", "pips", "atr"):
        result["sl_tp_mode"] = "percent"

    result["stop_loss_pct"] = max(0.1, min(50.0, float(result["stop_loss_pct"])))
    result["take_profit_pct"] = max(0.1, min(100.0, float(result["take_profit_pct"])))
    result["stop_loss_pips"] = max(0.1, min(2000.0, float(result["stop_loss_pips"])))
    result["take_profit_pips"] = max(0.1, min(4000.0, float(result["take_profit_pips"])))
    result["atr_period"] = max(2, min(100, int(result["atr_period"])))
    result["atr_sl_multiplier"] = max(0.1, min(10.0, float(result["atr_sl_multiplier"])))
    result["atr_tp_multiplier"] = max(0.1, min(20.0, float(result["atr_tp_multiplier"])))

    result["use_trailing_stop"] = bool(result["use_trailing_stop"])
    result["trailing_activation_pct"] = max(0.05, min(50.0, float(result["trailing_activation_pct"])))
    result["trailing_distance_pct"] = max(0.05, min(50.0, float(result["trailing_distance_pct"])))

    result["symbol"] = str(result.get("symbol") or "").strip()

    return result


def parse_strategy_text(description: str) -> dict:
    """Envia a descrição do cliente à IA e retorna os parâmetros estruturados.

    Levanta RuntimeError em caso de falha (chave ausente, erro de API, etc.).
    """
    if not description or not description.strip():
        raise RuntimeError("Descreva sua estratégia antes de interpretar.")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada. Defina a variável de ambiente ou "
            "adicione em .streamlit/secrets.toml."
        )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=PARSER_MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description.strip()}],
            output_config={"format": {"type": "json_schema", "schema": STRATEGY_SCHEMA}},
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Chave de API da Anthropic inválida.")
    except anthropic.RateLimitError:
        raise RuntimeError("Limite de requisições da API da Anthropic atingido. Tente novamente em instantes.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Erro na API da Anthropic ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Falha de conexão com a API da Anthropic. Verifique sua internet.")

    if response.stop_reason == "refusal":
        raise RuntimeError("A IA não conseguiu interpretar esta descrição.")

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise RuntimeError("A IA não retornou uma interpretação válida.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("A IA retornou uma resposta em formato inesperado.")

    return _clamp(parsed)
