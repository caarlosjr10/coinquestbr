"""
integrations/telegram_bot.py — Bot privado de alertas via API do Telegram.

Recurso exclusivo do Plano VIP. Usa a API HTTP de Bots do Telegram diretamente
via `requests` (sem dependências adicionais). O usuário fornece seu próprio
Bot Token e Chat ID, obtidos via @BotFather e @userinfobot (ver README.md).
"""

import requests

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
REQUEST_TIMEOUT = 10  # segundos


def send_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    """Envia uma mensagem de texto (Markdown) para o chat informado.

    Retorna (sucesso, mensagem_de_status).
    """
    if not bot_token or not chat_id:
        return False, "Bot Token e Chat ID são obrigatórios."

    url = TELEGRAM_API_URL.format(token=bot_token)
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return False, f"Falha de conexão com o Telegram: {e}"

    try:
        data = response.json()
    except ValueError:
        return False, f"Resposta inesperada do Telegram (HTTP {response.status_code})."

    if response.status_code == 200 and data.get("ok"):
        return True, "Mensagem enviada com sucesso."

    return False, data.get("description", f"Erro HTTP {response.status_code}")


def send_test_alert(bot_token: str, chat_id: str) -> tuple[bool, str]:
    """Dispara uma mensagem de teste formatada, para validar a configuração do bot."""
    text = (
        "🧪 *Teste de Alerta — CoinQuestBR*\n\n"
        "Este é um alerta de teste do seu Bot Privado.\n"
        "Se você recebeu esta mensagem, sua integração está funcionando corretamente\\! ✅"
    )
    return send_message(bot_token, chat_id, text)


def send_signal_alert(bot_token: str, chat_id: str, asset: str, signal_type: str, price: float, reason: str) -> tuple[bool, str]:
    """Dispara um alerta de sinal de trading (Compra/Venda) formatado com emoji.

    `signal_type` deve ser "Compra" ou "Venda".
    """
    emoji = "🟢" if signal_type.lower() == "compra" else "🔴"
    text = (
        f"{emoji} *Sinal de {signal_type.upper()}* — {asset}\n\n"
        f"💰 Preço: {price:,.4f}\n"
        f"📋 Motivo: {reason}\n\n"
        f"_Gerado por CoinQuestBR — não constitui recomendação de investimento._"
    )
    return send_message(bot_token, chat_id, text)
