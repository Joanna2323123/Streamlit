import streamlit as st
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
from io import StringIO

# Importações necessárias do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. MUDANÇA: Configuração da Página Focada em Análise Fiscal ---
st.set_page_config(
    page_title="Analisador Fiscal (NEXUS Básico)", # MUDADO
    page_icon="🧾", # MUDADO
    layout="wide"
)

# --- 2. MUDANÇA: Título e Descrição Claros ---
st.title("🧾 Analisador de Dados Fiscais (Versão Básica)")
st.write(
    "Faça o upload do seu `.zip` com arquivos CSV de notas fiscais. "
    "O agente Gemini irá analisar os dados e responder suas perguntas de negócio."
)

# --- 3. MUDANÇA: Mover o Upload para a Barra Lateral (Layout mais limpo) ---
with st.sidebar:
    st.header("Configuração")
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo .zip",
        type="zip"
    )
    
    # Adicionando uma nota sobre o tipo de dado esperado
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


# --- Lógica de Upload (Movido o selectbox para a sidebar) ---
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
                # Mover o selectbox para a sidebar também
                with st.sidebar:
                    selected_csv = st.selectbox("Selecione um arquivo CSV para analisar:", csv_files)
                
                if selected_csv:
                    st.session_state.selected_csv = selected_csv
                    with zip_ref.open(selected_csv) as f:
                        stringio = StringIO(f.read().decode('utf-8'))
                        st.session_state.df = pd.read_csv(stringio)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.session_state.df = None
else:
    # Mensagem de estado inicial se nenhum arquivo for carregado
    st.info("Por favor, faça o upload de um arquivo .zip na barra lateral para começar.")
    st.stop() # Não continua a execução se não houver arquivo

# --- Interação com o Agente (Só executa se o 'df' existir) ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_csv}' carregado. Pré-visualização dos dados:")
    st.dataframe(st.session_state.df) # Mantido st.dataframe() completo, como você alterou

    # --- 4. MUDANÇA: Adicionar Exemplos para guiar o usuário ---
    with st.expander("💡 Exemplos de perguntas que você pode fazer:"):
        st.markdown("""
        * Quais são os principais insights ou métricas de negócio?
        * Qual o Faturamento Total? (Precisa de uma coluna de 'valor')
        * Qual o cliente com maior valor? (Precisa de colunas 'cliente' e 'valor')
        * Qual o ticket médio por nota?
        * Qual a transação mais frequente? Compra ou venda? (Precisa de uma coluna 'natureza_da_operação' ou 'tipo')
        * Me dê um gráfico de pizza dos 5 setores mais comuns. (Precisa de uma coluna 'setor')
        * Qual a composição tributária (ICMS, PIS, COFINS) do cliente "Cliente X"?
        """)

    # --- 5. MUDANÇA: Placeholder do input focado em finanças ---
    user_question = st.text_input(
        "❓ Faça uma pergunta sobre seus dados fiscais:", # MUDADO
        placeholder="Qual o faturamento total?" # MUDADO
    )

    if user_question:
        with st.spinner("O Agente Gemini está pensando..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", # Mantido 'flash' para velocidade e quota
                    temperature=0,
                    google_api_key=google_api_key
                )
                
                # --- 6. MUDANÇA CRÍTICA: O AGENT_PREFIX ---
                # Trocado o prefixo de estatística pelo prefixo NEXUS/Fiscal
                AGENT_PREFIX = """
                Você é o "NEXUS", um agente especialista em análise de dados Fiscais e Financeiros. Seja direto, mas também robusto em suas respostas.

                **SUAS REGRAS DE COMPORTAMENTO:**

                1.  **VERIFICAÇÃO DE COLUNAS (REGRA MAIS IMPORTANTE):**
                    * **ANTES** de tentar responder a uma pergunta que exige colunas específicas (como 'ICMS', 'PIS', 'COFINS', 'cliente', 'valor_total', 'natureza_da_operação'), **PRIMEIRO** verifique se essas colunas existem em `df.columns`.
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

                st.success("Resposta do Agente:")
                st.write(output_text)
                
                fig = plt.gcf()
                if len(fig.get_axes()) > 0:
                    st.write("---")
                    st.subheader("📊 Gráfico Gerado")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")



