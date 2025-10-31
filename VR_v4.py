import streamlit as st
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
from io import StringIO

# Importações necessárias do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Configuração da Página do Streamlit ---
st.set_page_config(
    page_title="Analisador de CSV com Gemini",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 Análise de Dados com Agente Gemini")
st.write(
    "Faça o upload de um arquivo `.zip` contendo um ou mais CSVs. "
    "O agente usará o modelo Gemini do Google para responder perguntas sobre seus dados e gerar visualizações."
)

# --- Configuração da Chave de API (Método Seguro) ---
# O código tentará buscar a chave do sistema de segredos do Streamlit.
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Por favor, configure-a nos 'Secrets' do seu aplicativo no Streamlit Cloud.")
    st.stop()


# --- Lógica de Upload e Seleção de Arquivo ---
uploaded_file = st.file_uploader(
    "Faça o upload de um arquivo .zip",
    type="zip"
)

# Usando o estado da sessão para manter o dataframe carregado
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
                selected_csv = st.selectbox("Selecione um arquivo CSV para analisar:", csv_files)
                if selected_csv:
                    st.session_state.selected_csv = selected_csv
                    with zip_ref.open(selected_csv) as f:
                        stringio = StringIO(f.read().decode('utf-8'))
                        st.session_state.df = pd.read_csv(stringio)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.session_state.df = None

# --- Interação com o Agente ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_csv}' carregado. Visualizando as 5 primeiras linhas:")
    st.dataframe(st.session_state.df.head())

    user_question = st.text_input(
        "❓ Faça uma pergunta sobre os dados:",
        placeholder="Qual a correlação entre as variáveis?"
    )

    if user_question:
        with st.spinner("O Agente Gemini está pensando... 🧠"):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=0,
                    google_api_key=google_api_key
                )

                # --- INSTRUÇÕES AVANÇADAS PARA O AGENTE ANALISTA ---
                # --- INSTRUÇÕES AVANÇADAS PARA O AGENTE (MAIS DIRETO - VERSÃO NEXUS) ---
                AGENT_PREFIX = """
                Você é o "NEXUS", um agente especialista em análise de dados Fiscais e Financeiros. Seja direto, mas também robusto em suas respostas.

                **SUAS REGRAS DE COMPORTAMENTO:**

                1.  **VERIFICAÇÃO DE COLUNAS (REGRA MAIS IMPORTANTE):**
                    * **ANTES** de tentar responder a uma pergunta que exige colunas específicas (como 'ICMS', 'PIS', 'COFINS', 'cliente', 'valor_total', 'destinatario'), **PRIMEIRO** verifique se essas colunas existem no dataframe (`df.columns`).
                    * Se as colunas **NÃO EXISTIREM**, sua resposta **DEVE** ser informar ao usuário quais colunas estão faltando para aquela análise.
                    * **Exemplo de Resposta de Falha:** "Não posso calcular a composição tributária porque as colunas 'ICMS', 'PIS' e 'COFINS' não foram encontradas nos dados carregados."
                    * **NUNCA** falhe em silêncio ou retorne uma resposta vazia.

                2.  **PERGUNTAS GENÉRICAS (Se as colunas existirem):**
                    * Se o usuário fizer uma pergunta genérica ("Quais os principais dados?", "resumo") E as colunas de valor/cliente existirem, calcule as métricas NEXUS:
                        -   "Faturamento Total: [calculado]"
                        -   "Cliente de Maior Valor: [calculado]"
                        -   "Ticket Médio: [calculado]"
                    * Se as colunas para as métricas NEXUS não existirem, informe o usuário (Regra 1).
                    * **NÃO FAÇA** a análise de frequência de todas as colunas, a menos que o usuário peça (ex: "frequência por setor").

                3.  **PERGUNTAS ESPECÍFICAS (Se as colunas existirem):**
                    * Se o usuário perguntar sobre "distribuição" (ex: "valor por CFOP"), gere um gráfico de barras ou pizza.
                    * Se o usuário perguntar sobre "correlação", gere um heatmap.

                4.  **TOM DA RESPOSTA:**
                    * Seja um analista, não um assistente de chat.
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

                st.success("✅ Resposta do Agente:")
                st.write(output_text)
                
                fig = plt.gcf()
                if len(fig.get_axes()) > 0:
                    st.write("---")
                    st.subheader("📊 Gráfico Gerado")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")
else:
    st.info("Aguardando o upload de um arquivo .zip para iniciar a análise.")



