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

uploaded_files = st.file_uploader(
    "Envie um ou mais arquivos (.zip, .csv ou .pdf)",
    type=["zip", "csv", "pdf"],
    accept_multiple_files=True
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_csv' not in st.session_state:
    st.session_state.selected_csv = ""

if uploaded_files:
    try:
        # Se o usuário enviar vários PDFs, vamos iterar por eles
        pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
        zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
        csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]

        # 🔹 ZIP (permanece igual)
        if zip_files:
            uploaded_file = zip_files[0]
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                csv_inside = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_inside:
                    selected_csv = st.selectbox("Selecione um CSV dentro do ZIP:", csv_inside)
                    if selected_csv:
                        with zip_ref.open(selected_csv) as f:
                            stringio = StringIO(f.read().decode('utf-8'))
                            st.session_state.df = pd.read_csv(stringio)
                            st.session_state.selected_csv = selected_csv

        # 🔹 CSV individual
        elif csv_files:
            uploaded_file = csv_files[0]
            st.session_state.selected_csv = uploaded_file.name
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            st.session_state.df = pd.read_csv(stringio)

        # 🔹 Vários PDFs (novo comportamento)
        elif pdf_files:
            st.subheader("📄 PDFs carregados:")
            for pdf in pdf_files:
                st.write(f"- {pdf.name}")
            st.session_state.df = None
            st.info("PDFs carregados — perguntas textuais podem ser feitas ao modelo Gemini (sem dataframe).")

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.session_state.df = None

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



