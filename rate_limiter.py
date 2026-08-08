"""
rate_limiter.py — Controle de limites mensais de uso por usuário (Análises de
IA e Backtests), por plano.

MVP: persistência simples em arquivo JSON local, identificando o usuário pelo
e-mail informado na interface. Isso é suficiente para uso individual e para
demonstração do fluxo Grátis vs. Pro.

LIMITAÇÃO CONHECIDA (Streamlit Community Cloud): o sistema de arquivos do
container é efêmero — o arquivo de uso é preservado enquanto o app estiver
"quente", mas pode ser resetado após um redeploy ou período de inatividade
prolongado (o app "dorme" e reinicia). Para uso em produção com múltiplos
usuários reais, substitua esta camada por um banco de dados (ex: Supabase,
Firebase, ou uma tabela em Postgres).
"""

import json
import os
from datetime import datetime

USAGE_FILE = "usage_data.json"

# "ai_parecer" = geração de parecer técnico com IA; "backtest" = clique em
# "Analisar Estratégia" (interpretação + backtest). Cada um tem seu próprio
# contador mensal por e-mail.
AI_PARECER_LIMITS = {
    "Grátis": 1,
    "Pro": 50,
    "VIP": 500,
}

BACKTEST_LIMITS = {
    "Grátis": 2,
    "Pro": 500,
    "VIP": 2000,
}

_LIMITS_BY_KIND = {"ai_parecer": AI_PARECER_LIMITS, "backtest": BACKTEST_LIMITS}

# Mantido por compatibilidade com código antigo que importava PLAN_LIMITS.
PLAN_LIMITS = AI_PARECER_LIMITS


def _current_month_key() -> str:
    return datetime.now().strftime("%Y-%m")


def _load_usage() -> dict:
    if not os.path.exists(USAGE_FILE):
        return {}
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_usage(data: dict) -> None:
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_usage_count(user_id: str, kind: str = "ai_parecer") -> int:
    """Retorna quantas vezes o usuário já usou essa funcionalidade no mês corrente."""
    data = _load_usage()
    kind_data = data.get(user_id, {}).get(kind, {})

    if kind_data.get("month") != _current_month_key():
        return 0

    return kind_data.get("count", 0)


def check_limit(user_id: str, plan: str, kind: str = "ai_parecer") -> tuple[bool, int, int]:
    """Verifica se o usuário ainda tem uso disponível no mês para essa funcionalidade.

    Retorna (permitido, usado, limite).
    """
    limits = _LIMITS_BY_KIND.get(kind, AI_PARECER_LIMITS)
    limit = limits.get(plan, limits["Grátis"])
    used = get_usage_count(user_id, kind)
    return used < limit, used, limit


def increment_usage(user_id: str, kind: str = "ai_parecer") -> int:
    """Incrementa o contador de uso do usuário para o mês corrente e retorna o novo total."""
    data = _load_usage()
    month = _current_month_key()
    user_bucket = data.get(user_id, {})
    kind_data = user_bucket.get(kind, {})

    if kind_data.get("month") != month:
        kind_data = {"month": month, "count": 0}

    kind_data["count"] = kind_data.get("count", 0) + 1
    user_bucket[kind] = kind_data
    data[user_id] = user_bucket
    _save_usage(data)

    return kind_data["count"]
