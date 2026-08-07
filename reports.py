"""
reports.py — Geração de relatórios exportáveis (HTML e PDF) do backtest.

Recurso exclusivo do Plano VIP: reúne estatísticas, a curva de patrimônio e o
parecer técnico da IA em um único arquivo para download.
"""

import io
from datetime import datetime

import plotly.graph_objects as go
from fpdf import FPDF
from jinja2 import Template

HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Relatório CoinQuestBR — {{ symbol }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #1a1a1a; }
  h1 { color: #2E86C1; margin-bottom: 4px; }
  .subtitle { color: #666; margin-top: 0; }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }
  .metric-card { background: #f5f7fa; border-radius: 8px; padding: 16px; text-align: center; }
  .metric-card .value { font-size: 24px; font-weight: bold; color: #2E86C1; }
  .metric-card .label { font-size: 13px; color: #666; margin-top: 4px; }
  .ai-report { background: #fffbe6; border-left: 4px solid #f0c419; padding: 16px 20px; border-radius: 4px; white-space: pre-wrap; }
  .footer { margin-top: 40px; font-size: 12px; color: #999; border-top: 1px solid #eee; padding-top: 12px; }
</style>
</head>
<body>
  <h1>📈 CoinQuestBR — Relatório de Backtest</h1>
  <p class="subtitle">{{ symbol }} — {{ strategy_desc }}</p>
  <p class="subtitle">Gerado em {{ generated_at }}</p>

  <div class="metrics">
    <div class="metric-card"><div class="value">{{ metrics.win_rate }}%</div><div class="label">Win Rate</div></div>
    <div class="metric-card"><div class="value">{{ metrics.profit_factor }}</div><div class="label">Profit Factor</div></div>
    <div class="metric-card"><div class="value">{{ metrics.max_drawdown_pct }}%</div><div class="label">Drawdown Máximo</div></div>
    <div class="metric-card"><div class="value">{{ metrics.payoff }}</div><div class="label">Payoff</div></div>
    <div class="metric-card"><div class="value">{{ metrics.expectancy }}</div><div class="label">Expectativa</div></div>
    <div class="metric-card"><div class="value">{{ metrics.net_profit_pct }}%</div><div class="label">Resultado Líquido</div></div>
  </div>

  <p><strong>Total de Trades:</strong> {{ metrics.total_trades }} &nbsp;|&nbsp; <strong>Capital Final:</strong> {{ "%.2f"|format(metrics.final_equity) }}</p>

  <h2>Curva de Patrimônio</h2>
  {{ equity_chart_html | safe }}

  <h2>🤖 Parecer Técnico com IA</h2>
  <div class="ai-report">{{ ai_report }}</div>

  <div class="footer">
    Este relatório foi gerado automaticamente pelo CoinQuestBR e não constitui recomendação de investimento.
    Resultados passados (backtest) não garantem resultados futuros.
  </div>
</body>
</html>
""")


def _equity_figure(equity_df) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["equity"],
        mode="lines", name="Patrimônio", line=dict(color="#2E86C1", width=2),
        fill="tozeroy", fillcolor="rgba(46, 134, 193, 0.1)",
    ))
    fig.update_layout(
        xaxis_title="Data", yaxis_title="Patrimônio",
        template="plotly_white", height=400, margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def generate_html_report(symbol: str, strategy_desc: str, metrics: dict, equity_df, ai_report: str) -> str:
    """Gera um relatório HTML autocontido (funciona offline, exceto pelo Plotly via CDN)."""
    fig = _equity_figure(equity_df)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return HTML_TEMPLATE.render(
        symbol=symbol,
        strategy_desc=strategy_desc,
        metrics=metrics,
        equity_chart_html=chart_html,
        ai_report=ai_report or "Parecer técnico não gerado nesta sessão.",
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
    )


def _sanitize(text: str) -> str:
    """fpdf2 (fontes core) cobre Latin-1 — o suficiente para PT-BR; substitui o restante."""
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(symbol: str, strategy_desc: str, metrics: dict, equity_df, ai_report: str) -> bytes:
    """Gera um relatório em PDF com estatísticas, gráfico da equity curve e parecer da IA."""
    fig = _equity_figure(equity_df)
    chart_png = fig.to_image(format="png", width=900, height=400, scale=2)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _sanitize("CoinQuestBR - Relatorio de Backtest"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 7, _sanitize(f"{symbol} - {strategy_desc}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, _sanitize(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Metricas", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    rows = [
        ("Win Rate", f"{metrics.get('win_rate')}%"),
        ("Profit Factor", str(metrics.get("profit_factor"))),
        ("Drawdown Maximo", f"{metrics.get('max_drawdown_pct')}%"),
        ("Payoff", str(metrics.get("payoff"))),
        ("Expectativa Matematica", str(metrics.get("expectancy"))),
        ("Resultado Liquido", f"{metrics.get('net_profit_pct')}%"),
        ("Total de Trades", str(metrics.get("total_trades"))),
        ("Capital Final", f"{metrics.get('final_equity'):.2f}"),
    ]
    for label, value in rows:
        pdf.cell(70, 7, _sanitize(label))
        pdf.cell(0, 7, _sanitize(value), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Curva de Patrimonio", new_x="LMARGIN", new_y="NEXT")
    pdf.image(io.BytesIO(chart_png), x=10, w=190, type="PNG")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Parecer Tecnico com IA", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    body = ai_report or "Parecer tecnico nao gerado nesta sessao."
    pdf.multi_cell(0, 6, _sanitize(body))

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(
        0, 5,
        _sanitize(
            "Este relatorio foi gerado automaticamente e nao constitui recomendacao de investimento. "
            "Resultados passados (backtest) nao garantem resultados futuros."
        ),
    )

    return bytes(pdf.output())
