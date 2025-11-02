import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from io import StringIO
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Dados com Gemini", layout="wide")
st.title("Análise de Dados com Agente Gemini")
st.write(
    "Faça o upload de arquivos `.zip`, `.csv`, `.xlsx`, `.xls` ou `.pdf`. "
    "O agente usará o modelo Gemini do Google para responder perguntas sobre seus dados e gerar visualizações."
)

# --- Chave de API ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Configure-a nos 'Secrets' do seu aplicativo Streamlit Cloud.")
    st.stop()

# --- Upload ---
uploaded_files = st.file_uploader(
    "Envie um ou mais arquivos (.zip, .csv, .xlsx, .xls ou .pdf)",
    type=["zip", "csv", "xlsx", "xls", "pdf"],
    accept_multiple_files=True
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = ""

# --- Processamento de Arquivos ---
if uploaded_files:
    try:
        pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
        zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
        csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]
        excel_files = [f for f in uploaded_files if f.name.endswith((".xls", ".xlsx"))]

        if zip_files:
            uploaded_file = zip_files[0]
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                csv_inside = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_inside:
                    selected_csv = st.selectbox("Selecione um CSV dentro do ZIP:", csv_inside)
                    if selected_csv:
                        with zip_ref.open(selected_csv) as f:
                            stringio = StringIO(f.read().decode('utf-8'))
                            st.session_state.df = pd.read_csv(stringio, low_memory=False)
                            st.session_state.selected_file = selected_csv

        elif csv_files:
            uploaded_file = csv_files[0]
            st.session_state.selected_file = uploaded_file.name
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            st.session_state.df = pd.read_csv(stringio, low_memory=False)

        elif excel_files:
            uploaded_file = excel_files[0]
            st.session_state.selected_file = uploaded_file.name
            st.session_state.df = pd.read_excel(uploaded_file, engine="openpyxl")

        elif pdf_files:
            st.info("📂 PDFs carregados — perguntas textuais podem ser feitas ao modelo Gemini (sem dataframe).")
            st.session_state.df = None

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.session_state.df = None

# --- Interação com o Agente Gemini ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_file}' carregado com sucesso.")
    st.dataframe(st.session_state.df.head(50))

    user_question = st.text_input(
        "Faça uma pergunta sobre os dados:",
        placeholder="Exemplo: Quais foram os maiores pagamentos? Mostre como tabela."
    )

    if user_question:
        with st.spinner("O Agente Gemini está analisando os dados..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )

                AGENT_PREFIX = """
                Você é um Cientista Contábil e Administrador Financeiro. 
                Analise planilhas de dados financeiros, fiscais ou empresariais em profundidade.

                🔹 Regras:
                1. Quando o usuário pedir "tabela", "listagem" ou "exemplo tabular", retorne em formato de tabela pandas (df.head(), df.groupby(), etc.).
                2. Use `to_markdown()` apenas se a tabela for textual; caso contrário, entregue um DataFrame.
                3. Para gráficos, use matplotlib (histograma, barras, linha, pizza).
                4. Seja técnico, objetivo e contábil — como um analista de custos ou gestor financeiro.
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
                output_text = response.get("output", "")

                # --- NOVO: Detecção e exibição automática de tabelas ---
                try:
                    # tenta converter a resposta textual em DataFrame se parecer uma tabela
                    if "|" in output_text and "---" in output_text:
                        table_df = pd.read_csv(StringIO(output_text.replace("|", ",")))
                        st.write("📊 **Tabela Gerada:**")
                        st.dataframe(table_df)
                    else:
                        st.success("📊 Resposta do Agente:")
                        st.write(output_text)
                except Exception:
                    st.success("📊 Resposta do Agente:")
                    st.write(output_text)

                fig = plt.gcf()
                if len(fig.get_axes()) > 0:
                    st.write("---")
                    st.subheader("📈 Visualização Gerada")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")

elif uploaded_files and any(f.name.endswith(".pdf") for f in uploaded_files):
    user_question = st.text_input(
        "Pergunte algo sobre o texto dos PDFs:",
        placeholder="Exemplo: Resuma o conteúdo dos documentos fiscais."
    )
    if user_question:
        with st.spinner("O Agente está analisando o PDF..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )

                full_text = ""
                for pdf in [f for f in uploaded_files if f.name.endswith(".pdf")]:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        full_text += page.extract_text() or ""

                response = llm.invoke(f"Responda com base neste texto:\n{full_text}\n\nPergunta: {user_question}")
                st.success("📘 Resposta do Agente:")
                st.write(response.content)
            except Exception as e:
                st.error(f"Erro ao processar o PDF: {e}")

else:
    st.info("Aguardando o upload de um arquivo (.zip, .csv, .xlsx, .xls ou .pdf) para iniciar a análise.")




