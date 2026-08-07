"""
subscriptions.py — Controle de planos ativos por e-mail e captura de leads.

MVP sem gateway de pagamento integrado ainda: quando o cliente clica em
"Assinar" na página de Planos, o e-mail dele é salvo como um "lead" (interesse
de compra). O plano só é efetivamente ativado quando o administrador (você)
confere o pagamento (ex: Pix manual) e concede o plano pela página Admin.

Assim que um gateway de pagamento for integrado, a função `grant_plan` pode
ser chamada automaticamente a partir do webhook de confirmação de pagamento,
sem precisar mudar nada no resto do app.

LIMITAÇÃO CONHECIDA (Streamlit Community Cloud): assim como em
`rate_limiter.py`, a persistência é um arquivo JSON local, que é efêmero
entre reinicializações do app. Para produção com clientes reais, substitua
por um banco de dados (Supabase, Firebase, Postgres, etc.).
"""

import json
import os
from datetime import datetime

SUBSCRIPTIONS_FILE = "subscriptions.json"
LEADS_FILE = "leads.json"

VALID_PLANS = ["Grátis", "Pro", "VIP"]
DEFAULT_PLAN = "Grátis"


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Planos ativos
# ---------------------------------------------------------------------------

def get_plan_for_email(email: str) -> str:
    """Retorna o plano ativo do e-mail informado (Grátis se não encontrado)."""
    if not email:
        return DEFAULT_PLAN

    subs = _load(SUBSCRIPTIONS_FILE)
    entry = subs.get(email.strip().lower())
    if entry and entry.get("plan") in VALID_PLANS:
        return entry["plan"]
    return DEFAULT_PLAN


def grant_plan(email: str, plan: str, note: str = "") -> None:
    """Concede (ou atualiza) um plano para o e-mail informado."""
    if plan not in VALID_PLANS:
        raise ValueError(f"Plano inválido: {plan}")

    subs = _load(SUBSCRIPTIONS_FILE)
    subs[email.strip().lower()] = {
        "plan": plan,
        "granted_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    _save(SUBSCRIPTIONS_FILE, subs)


def revoke_plan(email: str) -> None:
    """Remove a assinatura do e-mail informado (volta para Grátis)."""
    subs = _load(SUBSCRIPTIONS_FILE)
    subs.pop(email.strip().lower(), None)
    _save(SUBSCRIPTIONS_FILE, subs)


def list_subscriptions() -> dict:
    """Retorna todas as assinaturas ativas: {email: {plan, granted_at, note}}."""
    return _load(SUBSCRIPTIONS_FILE)


# ---------------------------------------------------------------------------
# Leads (interesse de compra capturado na página de Planos)
# ---------------------------------------------------------------------------

def add_lead(email: str, plan: str) -> None:
    """Registra o interesse de um cliente em assinar um plano."""
    leads = _load(LEADS_FILE)
    entries = leads.get("entries", [])
    entries.append({
        "email": email.strip().lower(),
        "plan": plan,
        "requested_at": datetime.now().isoformat(timespec="seconds"),
    })
    leads["entries"] = entries
    _save(LEADS_FILE, leads)


def list_leads() -> list:
    """Retorna a lista de leads capturados, mais recentes primeiro."""
    leads = _load(LEADS_FILE)
    entries = leads.get("entries", [])
    return list(reversed(entries))
