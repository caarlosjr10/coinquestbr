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
    "natural escrita pelo usuário e decida se ela pode ser razoavelmente expressada por "
    "um motor de backtest que suporta APENAS: cruzamento de médias móveis (SMA ou EMA) "
    "com dois períodos, um filtro opcional de RSI (sobrevenda/sobrecompra), Stop Loss % "
    "e Take Profit %. "
    "\n\n"
    "Se a essência da estratégia puder ser mapeada para esses parâmetros (mesmo que de "
    "forma aproximada — ex: o usuário só descreveu a entrada e não o stop, então você "
    "completa com um valor padrão razoável), defina 'is_supported' como true e preencha "
    "todos os parâmetros numéricos. "
    "\n\n"
    "Se a estratégia depender fundamentalmente de conceitos que este motor NÃO suporta "
    "(ex: estrutura de mercado / Smart Money Concepts como BOS, CHoCH, order blocks, "
    "varredura de liquidez, padrões gráficos, price action discricionário, MACD, Bandas "
    "de Bollinger, volume, múltiplos timeframes, etc.) e não há como representar a "
    "intenção do usuário só com cruzamento de médias + RSI + SL/TP, defina "
    "'is_supported' como false. Nesse caso ainda preencha os campos numéricos com os "
    "valores padrão (EMA 9/21, sem RSI, SL 2%, TP 4% — eles serão ignorados pela "
    "aplicação). "
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
    "rapidamente, em 1 frase, como reformular a estratégia usando apenas médias móveis, "
    "RSI, stop e alvo.\n"
    "- Nunca escreva um parágrafo longo. Responda como se fosse uma mensagem de chat, "
    "curta e direta."
)

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_supported": {"type": "boolean"},
        "ma_type": {"type": "string", "enum": ["SMA", "EMA"]},
        "fast_period": {"type": "integer"},
        "slow_period": {"type": "integer"},
        "use_rsi_filter": {"type": "boolean"},
        "rsi_period": {"type": "integer"},
        "rsi_oversold": {"type": "integer"},
        "rsi_overbought": {"type": "integer"},
        "stop_loss_pct": {"type": "number"},
        "take_profit_pct": {"type": "number"},
        "interpretation_notes": {"type": "string"},
    },
    "required": [
        "is_supported", "ma_type", "fast_period", "slow_period", "use_rsi_filter",
        "rsi_period", "rsi_oversold", "rsi_overbought",
        "stop_loss_pct", "take_profit_pct", "interpretation_notes",
    ],
    "additionalProperties": False,
}

_DEFAULTS = {
    "is_supported": False,
    "ma_type": "EMA",
    "fast_period": 9,
    "slow_period": 21,
    "use_rsi_filter": False,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "interpretation_notes": "",
}


def _clamp(parsed: dict) -> dict:
    """Aplica limites de segurança (o JSON Schema da API não suporta min/max)."""
    result = {**_DEFAULTS, **parsed}

    result["fast_period"] = max(2, min(200, int(result["fast_period"])))
    result["slow_period"] = max(3, min(400, int(result["slow_period"])))
    if result["fast_period"] >= result["slow_period"]:
        result["slow_period"] = result["fast_period"] + 5

    result["rsi_period"] = max(2, min(100, int(result["rsi_period"])))
    result["rsi_oversold"] = max(1, min(50, int(result["rsi_oversold"])))
    result["rsi_overbought"] = max(50, min(99, int(result["rsi_overbought"])))

    result["stop_loss_pct"] = max(0.1, min(50.0, float(result["stop_loss_pct"])))
    result["take_profit_pct"] = max(0.1, min(100.0, float(result["take_profit_pct"])))

    if result["ma_type"] not in ("SMA", "EMA"):
        result["ma_type"] = "EMA"

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
