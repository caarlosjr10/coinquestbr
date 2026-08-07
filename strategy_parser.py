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
MAX_OUTPUT_TOKENS = 500

SYSTEM_PROMPT = (
    "Você é um interpretador de estratégias de trading. Leia a descrição em linguagem "
    "natural escrita pelo usuário e decida qual das DUAS famílias de motor de backtest "
    "melhor representa a estratégia dele — o motor NÃO executa código arbitrário, ele só "
    "sabe rodar essas duas famílias, então sua tarefa é encaixar a estratégia numa delas "
    "da forma mais fiel possível.\n"
    "\n"
    "FAMÍLIA 'classic' — cruzamento de médias móveis (SMA ou EMA) com dois períodos, "
    "filtro opcional de RSI (sobrevenda/sobrecompra), Stop Loss % e Take Profit %. Use "
    "esta família sempre que a estratégia for baseada em médias móveis, osciladores como "
    "RSI, ou qualquer ideia simples de tendência/momentum que dê pra aproximar disso.\n"
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
    "com a família escolhida, e preencha os parâmetros numéricos correspondentes. Para a "
    "família 'smc', se o usuário mencionar uma proporção de risco:retorno (ex: 'RR 1:2', "
    "'risco retorno 1:3'), converta para 'stop_loss_pct' (use 1.0 como padrão razoável se "
    "não houver stop explícito) e 'take_profit_pct' = stop_loss_pct multiplicado pela "
    "proporção informada.\n"
    "\n"
    "Se a estratégia depender de algo que NENHUMA das duas famílias representa (ex: "
    "dados fundamentalistas/notícias, múltiplos timeframes combinados, MACD, Bandas de "
    "Bollinger, volume, padrões de candle isolados sem estrutura, ou qualquer coisa vaga "
    "demais para virar regras objetivas), defina 'is_supported' como false. Nesse caso "
    "ainda preencha todos os campos numéricos com os valores padrão (serão ignorados "
    "pela aplicação).\n"
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
    "- Se is_supported=true: confirme rapidamente o que você entendeu, e só se algo "
    "precisou ser aproximado/simplificado, diga isso em poucas palavras — sem jargão "
    "técnico, sem explicar o motivo em detalhe.\n"
    "- Se is_supported=false: diga em 1 frase o que não deu para configurar e sugira "
    "rapidamente, em 1 frase, como reformular a estratégia usando médias móveis/RSI ou "
    "os passos de varredura+BOS+pullback+CHoCH.\n"
    "- Nunca escreva um parágrafo longo. Responda como se fosse uma mensagem de chat, "
    "curta e direta."
)

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_supported": {"type": "boolean"},
        "strategy_family": {"type": "string", "enum": ["classic", "smc"]},
        "ma_type": {"type": "string", "enum": ["SMA", "EMA"]},
        "fast_period": {"type": "integer"},
        "slow_period": {"type": "integer"},
        "use_rsi_filter": {"type": "boolean"},
        "rsi_period": {"type": "integer"},
        "rsi_oversold": {"type": "integer"},
        "rsi_overbought": {"type": "integer"},
        "swing_strength": {"type": "integer"},
        "max_setup_bars": {"type": "integer"},
        "stop_loss_pct": {"type": "number"},
        "take_profit_pct": {"type": "number"},
        "symbol": {"type": "string"},
        "interpretation_notes": {"type": "string"},
    },
    "required": [
        "is_supported", "strategy_family", "ma_type", "fast_period", "slow_period",
        "use_rsi_filter", "rsi_period", "rsi_oversold", "rsi_overbought",
        "swing_strength", "max_setup_bars",
        "stop_loss_pct", "take_profit_pct", "symbol", "interpretation_notes",
    ],
    "additionalProperties": False,
}

_DEFAULTS = {
    "is_supported": False,
    "strategy_family": "classic",
    "ma_type": "EMA",
    "fast_period": 9,
    "slow_period": 21,
    "use_rsi_filter": False,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "swing_strength": 2,
    "max_setup_bars": 40,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "symbol": "",
    "interpretation_notes": "",
}


def _clamp(parsed: dict) -> dict:
    """Aplica limites de segurança (o JSON Schema da API não suporta min/max)."""
    result = {**_DEFAULTS, **parsed}

    if result["strategy_family"] not in ("classic", "smc"):
        result["strategy_family"] = "classic"

    result["fast_period"] = max(2, min(200, int(result["fast_period"])))
    result["slow_period"] = max(3, min(400, int(result["slow_period"])))
    if result["fast_period"] >= result["slow_period"]:
        result["slow_period"] = result["fast_period"] + 5

    result["rsi_period"] = max(2, min(100, int(result["rsi_period"])))
    result["rsi_oversold"] = max(1, min(50, int(result["rsi_oversold"])))
    result["rsi_overbought"] = max(50, min(99, int(result["rsi_overbought"])))

    result["swing_strength"] = max(1, min(10, int(result["swing_strength"])))
    result["max_setup_bars"] = max(5, min(200, int(result["max_setup_bars"])))

    result["stop_loss_pct"] = max(0.1, min(50.0, float(result["stop_loss_pct"])))
    result["take_profit_pct"] = max(0.1, min(100.0, float(result["take_profit_pct"])))

    if result["ma_type"] not in ("SMA", "EMA"):
        result["ma_type"] = "EMA"

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
