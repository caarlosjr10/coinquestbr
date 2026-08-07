"""
rate_limiter.py — Controle de limite mensal de Análises de IA por usuário.

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

PLAN_LIMITS = {
    "Grátis": 1,
    "Pro": 50,
    "VIP": 500,
}


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


def get_usage_count(user_id: str) -> int:
    """Retorna quantas análises de IA o usuário já fez no mês corrente."""
    data = _load_usage()
    user_data = data.get(user_id, {})

    if user_data.get("month") != _current_month_key():
        return 0

    return user_data.get("count", 0)


def check_limit(user_id: str, plan: str) -> tuple[bool, int, int]:
    """Verifica se o usuário ainda tem análises disponíveis no mês.

    Retorna (permitido, usado, limite).
    """
    limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["Grátis"])
    used = get_usage_count(user_id)
    return used < limit, used, limit


def increment_usage(user_id: str) -> int:
    """Incrementa o contador de uso do usuário para o mês corrente e retorna o novo total."""
    data = _load_usage()
    month = _current_month_key()
    user_data = data.get(user_id, {})

    if user_data.get("month") != month:
        user_data = {"month": month, "count": 0}

    user_data["count"] = user_data.get("count", 0) + 1
    data[user_id] = user_data
    _save_usage(data)

    return user_data["count"]
