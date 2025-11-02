import streamlit as st
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
from io import StringIO

# Importações necessárias do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. MUDANÇA: Configuração da Página ---
st.set_page_config(
    page_title="Analisador Fiscal (NEXUS Básico)", 
    page_icon="🧾",
    layout="wide"
    # O Streamlit usará o tema (dark/light) do sistema do usuário.
    # A imagem que você enviou tem um tema escuro, que será aplicado
    # se o sistema do usuário estiver em modo escuro.
)

# --- Título e Descrição (Sem alteração) ---
st.title("🧾 Analisador de Dados Fiscais (Versão Básica)")
st.write(
    "Faça o upload do seu `.zip` com arquivos CSV de notas fiscais. "
    "O agente Gemini irá analisar os dados e responder suas perguntas de negócio."
)

# --- Upload na Barra Lateral (Sem alteração) ---
with st.sidebar:
    st.header("Configuração")
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo .zip",
        type="zip"
    )
    
    st.info(
        "Este agente é otimizado para analisar dados fiscais. "
        "Ele funciona melhor com colunas como 'cliente', 'valor_total', 'ICMS', 'PIS', 'COFINS', 'natureza_da_operação', 'setor', etc."
    )

# --- Configuração da Chave de API (Sem alteração) ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Por favor, configure-a nos 'Secrets' do seu aplicativo no Streamlit Cloud.")
    st.stop()

# --- 2. MUDANÇA: Inicializar o histórico do chat ---
# Isso é essencial para o layout de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Lógica de Upload (Sem alteração na lógica, apenas no local) ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_csv' not in st.session_state:
    st.session_state.selected_csv = ""

if uploaded_file:
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                st.warning("O arquivo .zip não contém nenhum arquivo .csv.")
                st.session_state.df = None
            else:
                with st.sidebar:
                    selected_csv = st.selectbox("Selecione um arquivo CSV para analisar:", csv_files)
                
                if selected_csv:
                    # Se o usuário trocar o CSV, limpa o chat antigo
                    if st.session_state.selected_csv != selected_csv:
                        st.session_state.messages = []
                        
                    st.session_state.selected_csv = selected_csv
                    with zip_ref.open(selected_csv) as f:
                        stringio = StringIO(f.read().decode('utf-8'))
                        st.session_state.df = pd.read_csv(stringio)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.session_state.df = None
else:
    st.info("Por favor, faça o upload de um arquivo .zip na barra lateral para começar.")
    st.stop()

# --- 3. MUDANÇA: Lógica de Interação com o Agente (Totalmente Refatorada para Chat) ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_csv}' carregado. Pré-visualização dos dados:")
    st.dataframe(st.session_state.df) 

    with st.expander("💡 Exemplos de perguntas que você pode fazer:"):
        st.markdown("""
        * Quais são os principais insights ou métricas de negócio?
        * Qual o Faturamento Total?
        * Qual o cliente com maior valor?
        * Qual o ticket médio por nota?
        * Qual a transação mais frequente? Compra ou venda?
        * Me dê um gráfico de pizza dos 5 setores mais comuns.
        * Quais insights e oportunidades de negócios esses dados podem revelar?
        """)
    
    st.subheader("Chat Interativo com IA") # Título da imagem

    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Nota: Esta versão simples não re-exibe gráficos do histórico.
            # Apenas a resposta em texto é salva em st.session_state.

    # Nova entrada do usuário (caixa de chat no final da página)
    if user_question := st.chat_input("Pergunte sobre o relatório..."): # Placeholder da imagem
        
        # Adiciona e exibe a mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # Gera e exibe a resposta do Agente
        with st.chat_message("assistant"): # Balão do assistente (lado esquerdo)
            with st.spinner("O Agente Gemini está pensando..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash", 
                        temperature=0,
                        google_api_key=google_api_key
                    )
                    
                    # Cérebro do Agente (NEXUS) - Sem alteração
                    AGENT_PREFIX = """
                    Você é o "NEXUS", um agente especialista em análise de dados Fiscais e Financeiros. Seja direto, mas também robusto em suas respostas.

                    **SUAS REGRAS DE COMPORTAMENTO:**

                    1.  **VERIFICAÇÃO DE COLUNAS (REGRA MAIS IMPORTANTE):**
                        * **ANTES** de tentar responder a uma pergunta que exige colunas específicas (como 'ICMS', 'PIS', 'COFINS', 'cliente', 'valor_total', 'natureza_da_operação'), **PRIMEIRO** verifique if those columns exist in `df.columns`.
                        * Se as colunas **NÃO EXISTIREM**, sua resposta **DEVE** ser informar ao usuário quais colunas estão faltando para aquela análise.
                        * **Exemplo de Resposta de Falha:** "Não posso calcular. As colunas 'ICIS', 'PIS' e 'COFINS' não foram encontradas nos dados."
                        * **NÃO FALHE EM SILÊNCIO.**

                    2.  **PERGUNTAS GENÉRICAS (MÉTRICAS NEXUS):**
                        * Se o usuário fizer uma pergunta genérica ("Quais os principais dados?", "resumo", "métricas", "insights") E as colunas necessárias existirem, calcule as métricas de negócio principais:
                            - "Faturamento Total: [some a coluna de valor]"
                            - "Cliente de Maior Valor: [identifique o cliente com maior valor]"
                            - "Ticket Médio: [calcule o valor total / contagem de notas]"
                        * Se as colunas não existirem, informe o usuário (Regra 1).

                    3.  **PERGUNTAS ESPECÍFICAS (GRÁFICOS):**
                        * Se o usuário perguntar sobre "distribuição" ou "comparação" (ex: "valor por setor", "operações por tipo"), gere um gráfico de barras ou pizza.
                        * Se o usuário perguntar sobre "correlação", gere um heatmap.

                    4.  **TOM DA RESPOSTA:**
                        * Seja um analista de negócios, direto ao ponto.
                    """
                    
                    agent = create_pandas_dataframe_agent(
                        llm,
                        st.session_state.df,
                        prefix=AGENT_PREFIX,
                        verbose=False,
                        agent_type="openai-tools",
                        handle_parsing_errors=True,
                        allow_dangerous_code=True,
                    )
                    
                    plt.close('all')
                    response = agent.invoke({"input": user_question})
                    output_text = response.get("output", "Não foi possível gerar uma resposta.")

                    # Exibe a resposta em texto
                    st.write(output_text)
                    
                    # Adiciona a resposta (só texto) ao histórico
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                    # Exibe o gráfico, se houver, dentro da mesma bolha
                    fig = plt.gcf()
                    if len(fig.get_axes()) > 0:
                        st.pyplot(fig)

                except Exception as e:
                    error_message = f"Ocorreu um erro durante a execução do agente: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
