# CoinQuestBR 📈

Aplicativo web em Streamlit para **validação e backtest massivo de estratégias de trading**, com **parecer técnico-econômico gerado por IA** (Anthropic Claude), **exportação de relatórios** e **alertas via Telegram**.

## Funcionalidades

- **Mercados suportados:** Cripto (via `ccxt`) e Ações / Forex / Índices (via `yfinance`).
- **Estratégia em linguagem natural:** o cliente descreve a estratégia em texto livre (ex: "cruzamento de EMA 9/21 com RSI abaixo de 35, stop 2%, alvo 5%") e a IA interpreta e monta os parâmetros do backtest — cruzamento de Médias Móveis (SMA/EMA), filtro de RSI opcional, Stop Loss % e Take Profit %ª —, mostrando um resumo editável para confirmação antes de rodar.
- **Dois modos de execução:**
  - **Backtest Único** — roda a estratégia sobre N candles históricos (500 a 5000).
  - **Otimização (Grid Search)** — testa múltiplas combinações de parâmetros e retorna a melhor.
- **Dashboard de métricas:** Win Rate, Profit Factor, Drawdown Máximo, Payoff e Expectativa Matemática.
- **Curva de Patrimônio (Equity Curve)** interativa via Plotly.
- **Parecer Técnico com IA, anti-alucinação:** o modelo recebe **apenas** um bloco JSON estrito com as métricas do backtest e é instruído (via System Prompt) a nunca inventar dados, nunca prever lucros futuros e nunca dar conselho financeiro direto — se uma métrica não estiver disponível, ele diz isso explicitamente.
- **Sinal de entrada/saída no último candle**, com opção de disparo imediato de alerta no Telegram (VIP).
- **Exportação de relatório completo** (estatísticas + gráfico + parecer da IA) em **HTML** e **PDF** (VIP).
- **Página de Planos** com cards comerciais e captura de interesse de compra (checkout).
- **Painel Admin** (protegido por senha) para conceder planos manualmente enquanto não há gateway de pagamento integrado.
- **Controle de limites por plano** (definido automaticamente pelo e-mail do cliente, sem precisar escolher nada):
  - **Grátis**: 1 análise de IA por mês.
  - **Pro**: 50 análises de IA por mês.
  - **VIP / Institutional** (R$ 297/mês): 500 análises de IA por mês, processamento com modelo mais avançado (`claude-sonnet-5`), exportação de relatórios e bot de alertas no Telegram.
  - Ao atingir o limite, um aviso de upgrade é exibido.

## Arquitetura

```
coinquestbr/
├── app.py                          # Roteador do app multi-página (st.navigation)
├── views/
│   ├── backtest.py                 # Página principal: estratégia por texto, backtest, parecer, exportação, Telegram
│   ├── planos.py                   # Página de Planos: cards comerciais + captura de leads (checkout)
│   └── admin.py                    # Painel administrativo (senha): conceder planos, ver leads
├── engine.py                       # Dados (ccxt/yfinance), indicadores, backtest, otimização, sinal atual
├── ai_analysis.py                  # Integração com a API da Anthropic (anti-alucinação, seleção de modelo por plano)
├── strategy_parser.py              # IA interpreta a descrição em texto da estratégia -> parâmetros estruturados
├── rate_limiter.py                 # Controle de limite mensal de análises de IA por plano
├── subscriptions.py                # Planos ativos por e-mail + leads de interesse (checkout)
├── reports.py                      # Geração de relatórios exportáveis (HTML e PDF)
├── integrations/
│   └── telegram_bot.py             # Bot privado de alertas via API do Telegram
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example        # Modelo do arquivo de secrets (copiar e preencher)
└── .gitignore
```

> **Nota sobre persistência:** o controle de limites, as assinaturas e os leads usam arquivos JSON locais (`usage_data.json`, `subscriptions.json`, `leads.json`). No Streamlit Community Cloud o sistema de arquivos é efêmero entre reinicializações do app — para uso com múltiplos usuários reais em produção, substitua essas camadas por um banco de dados (Supabase, Firebase, Postgres, etc.). O mesmo vale para os campos de Bot Token/Chat ID do Telegram, que hoje ficam apenas na sessão do navegador.

> **Nota sobre pagamento:** este MVP ainda **não tem gateway de pagamento integrado**. A página de Planos captura o e-mail de quem quer assinar (lead); a ativação do plano é feita manualmente por você na página **Admin**, após confirmar o pagamento por fora (ex: Pix). Quando integrar um gateway (Mercado Pago, Stripe, etc.), basta chamar `subscriptions.grant_plan()` a partir do webhook de confirmação de pagamento — o resto do app não muda. O preço do plano **Pro (R$ 97/mês)** é um valor de exemplo — ajuste em `views/planos.py`.

---

## 1. Rodando localmente

### Pré-requisitos
- Python 3.10+
- Uma chave de API da Anthropic (obtida em [console.anthropic.com](https://console.anthropic.com))

### Passo a passo

```bash
# 1. Clone ou entre na pasta do projeto
cd coinquestbr

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure sua chave de API e a senha do Admin
# Copie o arquivo de exemplo:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edite .streamlit/secrets.toml: cole sua chave ANTHROPIC_API_KEY e escolha uma ADMIN_PASSWORD

# 5. Rode o app
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## 2. Publicando gratuitamente no Streamlit Community Cloud (24/7 online)

### Passo 1 — Subir o código no GitHub

```bash
# Dentro da pasta coinquestbr
git init
git add .
git commit -m "Initial commit: CoinQuestBR MVP"

# Crie um repositório novo no GitHub (via site github.com/new), depois:
git remote add origin https://github.com/SEU-USUARIO/coinquestbr.git
git branch -M main
git push -u origin main
```

> ⚠️ **Importante:** o arquivo `.streamlit/secrets.toml` (com sua chave real) está no `.gitignore` e **não** será enviado ao GitHub. Isso é proposital — a chave deve ser configurada diretamente no painel do Streamlit Cloud (passo 3 abaixo), nunca commitada no repositório.

### Passo 2 — Criar o app no Streamlit Community Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com sua conta GitHub.
2. Clique em **"New app"**.
3. Selecione o repositório `coinquestbr`, branch `main` e o arquivo principal `app.py`.
4. Clique em **"Deploy"**.

### Passo 3 — Configurar a chave da API (Secrets)

1. No painel do app, vá em **Settings → Secrets**.
2. Cole o conteúdo abaixo (substituindo pela sua chave real):

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ADMIN_PASSWORD = "escolha-uma-senha-forte"
```

3. Salve. O app reiniciará automaticamente com a chave disponível.

Pronto — o app ficará online gratuitamente, hospedado pelo Streamlit Community Cloud, acessível por uma URL pública (ex: `https://coinquestbr.streamlit.app`).

### Atualizando o app depois de mudanças

```bash
git add .
git commit -m "Descrição da mudança"
git push
```

O Streamlit Cloud detecta o push e faz redeploy automaticamente.

---

## 3. Gestão econômica de tokens da API e anti-alucinação

- **Grátis/Pro** usam **`claude-haiku-4-5`** — o modelo mais rápido e barato da linha atual da Anthropic.
- **VIP** usa **`claude-sonnet-5`** — modelo mais capaz, para um parecer mais aprofundado ("processamento prioritário"), com `thinking` desativado para manter custo e latência previsíveis numa tarefa curta.
- O prompt enviado à IA contém **apenas um bloco JSON estrito com as métricas do backtest** (Win Rate, Profit Factor, Drawdown, Payoff, Expectativa), nunca o histórico completo de trades — isso mantém o custo por chamada mínimo.
- **Camada anti-alucinação:** o System Prompt instrui explicitamente o modelo a usar somente os dados fornecidos, nunca inventar estatísticas/preços, nunca prever lucros futuros e nunca dar conselho financeiro direto. Métricas indisponíveis (ex: backtest sem nenhum trade) são enviadas como `null` em vez de `0`, e o modelo é instruído a relatar "não disponível" nesses casos.
- A resposta é limitada a poucos tokens (parecer de até 3 parágrafos curtos).
- O sistema de rate limiting (`rate_limiter.py`) impede uso excessivo por usuário, protegendo o orçamento de API mesmo em caso de abuso.

---

## 4. Configurando o Bot de Alertas do Telegram (Plano VIP)

O painel "🔔 Bot de Alertas do Telegram" (barra lateral, visível para assinantes VIP) precisa de um **Bot Token** e de um **Chat ID**. Veja como obter os dois:

### Passo 1 — Criar um bot com o @BotFather

1. No Telegram, procure por **[@BotFather](https://t.me/BotFather)** e inicie uma conversa.
2. Envie o comando `/newbot`.
3. Escolha um **nome de exibição** para o bot (ex: `Meu Alerta CoinQuest`).
4. Escolha um **username** único terminado em `bot` (ex: `meu_alerta_coinquest_bot`).
5. O @BotFather responderá com uma mensagem contendo o **Bot Token** — algo como:
   ```
   123456789:ABCdefGhIJKlmNoPQRstuVwxyZ
   ```
   Copie esse token — ele vai no campo **"Bot Token"** do app.

### Passo 2 — Obter seu Chat ID

1. Envie **qualquer mensagem** (ex: "Olá") para o bot que você acabou de criar (procure pelo username escolhido).
2. Procure por **[@userinfobot](https://t.me/userinfobot)** no Telegram e inicie uma conversa com ele — ele responderá com o seu **Chat ID** (um número, ex: `123456789`).
   - Alternativa: acesse `https://api.telegram.org/bot<SEU_TOKEN>/getUpdates` no navegador logo após enviar a mensagem ao bot, e procure pelo campo `"chat":{"id": ...}` na resposta JSON.
3. Copie esse número — ele vai no campo **"Chat ID"** do app.

### Passo 3 — Testar

1. Cole o **Bot Token** e o **Chat ID** nos campos do painel lateral do CoinQuestBR.
2. Clique em **"📨 Enviar Alerta de Teste"**.
3. Você deve receber uma mensagem de teste do seu bot no Telegram. Se não receber, confira se enviou uma mensagem ao bot primeiro (o Telegram só permite que bots iniciem conversas com usuários que já interagiram com eles).

Depois de configurado, ao rodar um backtest o app detecta se o **último candle** gerou um sinal de Compra/Venda e permite disparar esse alerta manualmente para o seu Telegram com um clique.
