"""
ai_analysis.py — Geração de parecer técnico-econômico via API da Anthropic.

Camada de segurança e anti-alucinação:
- O System Prompt instrui o modelo a usar EXCLUSIVAMENTE os dados fornecidos.
- Os dados são injetados como um bloco JSON estrito (fonte única da verdade).
- Métricas indisponíveis (ex: backtest sem trades) são explicitamente marcadas
  como `null`, para que o modelo relate "não disponível" em vez de inventar.

Otimização de custo:
- Grátis/Pro: `claude-haiku-4-5` — modelo mais rápido e barato da linha atual.
- VIP: `claude-sonnet-5` — modelo mais capaz, para pareceres mais aprofundados
  ("processamento prioritário"), com `thinking` desativado para manter o custo
  previsível numa tarefa curta.
- Resposta sempre limitada a poucos tokens (parecer de até 3 parágrafos).
"""

import json
import os

import anthropic
import streamlit as st

# Modelo padrão: rápido e barato, ideal para resumos curtos (planos Grátis e Pro).
DEFAULT_MODEL = "claude-haiku-4-5"

# Modelo do plano VIP: mais capaz, para uma análise mais aprofundada.
VIP_MODEL = "claude-sonnet-5"

MAX_OUTPUT_TOKENS = 500
VIP_MAX_OUTPUT_TOKENS = 800

SYSTEM_PROMPT = (
    "Você é um analista quantitativo focado exclusivamente em dados históricos de backtest. "
    "Você deve basear sua análise SOMENTE nos dados fornecidos na tabela JSON da mensagem do "
    "usuário — nunca utilize informações externas, conhecimento de mercado geral ou suposições. "
    "Nunca invente estatísticas, porcentagens ou preços que não foram explicitamente fornecidos "
    "na tabela. Não faça previsões de lucros futuros nem dê conselhos financeiros diretos "
    "(ex: 'compre', 'venda', 'invista'). Se um dado necessário não constar na tabela ou estiver "
    "como null, informe explicitamente que aquela métrica não está disponível em vez de estimá-la. "
    "Responda em português, em no máximo 3 parágrafos curtos, sem listas, sem markdown e sem títulos."
)


def _build_data_table(metrics: dict) -> dict:
    """Monta a tabela de dados estrita enviada ao modelo.

    Se não houve nenhum trade, as métricas derivadas (win rate, profit factor
    etc.) são marcadas como `null` em vez de 0.0 — um zero real seria
    indistinguível de "sem dados" e poderia induzir o modelo a uma leitura
    equivocada.
    """
    if not metrics.get("total_trades"):
        return {
            "total_trades": 0,
            "win_rate_pct": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "payoff": None,
            "expectancy": None,
            "net_profit_pct": None,
        }

    return {
        "total_trades": metrics.get("total_trades"),
        "win_rate_pct": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "payoff": metrics.get("payoff"),
        "expectancy": metrics.get("expectancy"),
        "net_profit_pct": metrics.get("net_profit_pct"),
    }


def _build_user_prompt(metrics: dict, strategy_desc: str, asset: str) -> str:
    data_table = _build_data_table(metrics)
    data_json = json.dumps(data_table, ensure_ascii=False, indent=2)

    return f"""Ativo: {asset}
Estratégia: {strategy_desc}

DADOS (JSON — única fonte de informação permitida; campos com valor null significam
métrica não disponível para este backtest):
{data_json}

Baseando-se exclusivamente nos dados acima, escreva um PARECER TÉCNICO ECONÔMICO em até 3 parágrafos curtos:
Parágrafo 1 — robustez estatística (número de trades, win rate vs. profit factor).
Parágrafo 2 — risco (drawdown máximo e payoff) e se o risco/retorno parece adequado.
Parágrafo 3 — veredito direto: a estratégia parece promissora, precisa de ajustes, ou não é
recomendada — e por quê, com base apenas nos dados fornecidos."""


def generate_analysis(metrics: dict, strategy_desc: str, asset: str, plan: str = "Grátis") -> str:
    """Chama a API da Anthropic e retorna o parecer técnico em texto puro.

    `plan` seleciona o modelo: usuários VIP recebem `claude-sonnet-5`
    (processamento prioritário / mais aprofundado); demais planos usam
    `claude-haiku-4-5` (mais barato e rápido).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada. Defina a variável de ambiente ou "
            "adicione em .streamlit/secrets.toml."
        )

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = _build_user_prompt(metrics, strategy_desc, asset)

    is_vip = plan == "VIP"
    model = VIP_MODEL if is_vip else DEFAULT_MODEL
    max_tokens = VIP_MAX_OUTPUT_TOKENS if is_vip else MAX_OUTPUT_TOKENS

    request_kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if is_vip:
        # Sonnet 5 roda com thinking adaptativo por padrão; para um parecer
        # curto e factual, desativamos para manter custo e latência previsíveis.
        request_kwargs["thinking"] = {"type": "disabled"}

    try:
        response = client.messages.create(**request_kwargs)
    except anthropic.AuthenticationError:
        raise RuntimeError("Chave de API da Anthropic inválida.")
    except anthropic.RateLimitError:
        raise RuntimeError("Limite de requisições da API da Anthropic atingido. Tente novamente em instantes.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Erro na API da Anthropic ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Falha de conexão com a API da Anthropic. Verifique sua internet.")

    if response.stop_reason == "refusal":
        raise RuntimeError("A IA recusou gerar o parecer para esta solicitação.")

    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n\n".join(text_blocks).strip()
