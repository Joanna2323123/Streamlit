import streamlit as st
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

# Para leitura de PDFs
from PyPDF2 import PdfReader

# Importações do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Configuração da Página ---
st.set_page_config(
    page_title="Analisador de Dados com Gemini",
    layout="wide"
)
st.title("Análise de Dados com Agente Gemini")
st.write(
    "Faça o upload de um arquivo `.zip`, `.csv` ou `.pdf`. "
    "O agente usará o modelo Gemini para responder perguntas sobre seus dados e gerar visualizações."
)

# --- Chave de API ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Configure-a nos 'Secrets' do Streamlit Cloud.")
    st.stop()

# --- Upload de Arquivo ---
uploaded_file = st.file_uploader(
    "Envie um arquivo (.zip, .csv ou .pdf)",
    type=["zip", "csv", "pdf"]
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_csv' not in st.session_state:
    st.session_state.selected_csv = ""

# --- Tratamento dos tipos de arquivo ---
if uploaded_file:
    try:
        file_name = uploaded_file.name

        # 1️⃣ Caso ZIP
        if file_name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    st.warning("O arquivo ZIP não contém CSVs.")
                    st.session_state.df = None
                else:
                    selected_csv = st.selectbox("Selecione um arquivo CSV para analisar:", csv_files)
                    if selected_csv:
                        st.session_state.selected_csv = selected_csv
                        with zip_ref.open(selected_csv) as f:
                            stringio = StringIO(f.read().decode('utf-8'))
                            st.session_state.df = pd.read_csv(stringio)

        # 2️⃣ Caso CSV individual
        elif file_name.endswith(".csv"):
            st.session_state.selected_csv = file_name
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            st.session_state.df = pd.read_csv(stringio)

        # 3️⃣ Caso PDF
        elif file_name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            st.text_area("📄 Conteúdo extraído do PDF:", text[:4000], height=300)
            st.session_state.df = None
            st.info("PDF carregado — perguntas textuais podem ser feitas ao modelo Gemini (sem dataframe).")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.session_state.df = None

# --- Interação com o Agente ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_csv}' carregado. Visualizando as 5 primeiras linhas:")
    st.dataframe(st.session_state.df)
    user_question = st.text_input(
        "Faça uma pergunta sobre os dados:",
        placeholder="Qual a correlação entre as variáveis?"
    )
    if user_question:
        with st.spinner("O Agente Gemini está pensando..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )

                AGENT_PREFIX = """
                Você é um agente especialista em análise de dados. Sua principal função é fornecer insights através de visualizações. 
                **Regras:**
                1. Para "valores frequentes", use value_counts() em colunas categóricas (<25 valores únicos).
                2. Para "variabilidade" ou "distribuição", use histograma e boxplot.
                3. Para "correlação", gere um heatmap.
                4. Sempre que possível, priorize gráficos ao texto.
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

elif uploaded_file and uploaded_file.name.endswith(".pdf"):
    user_question = st.text_input(
        "Pergunte algo sobre o texto do PDF:",
        placeholder="Resuma o conteúdo ou destaque tópicos importantes."
    )
    if user_question:
        with st.spinner("O Agente Gemini está analisando o PDF..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )
                reader = PdfReader(uploaded_file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                response = llm.invoke(f"Responda com base neste texto:\n{text}\n\nPergunta: {user_question}")
                st.success("Resposta do Agente:")
                st.write(response.content)
            except Exception as e:
                st.error(f"Erro ao processar o PDF: {e}")

else:
    st.info("Aguardando o upload de um arquivo (.zip, .csv ou .pdf) para iniciar a análise.")


