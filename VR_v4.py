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

        # ZIP
        if zip_files:
            uploaded_file = zip_files[0]
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                csv_inside = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_inside:
                    selected_csv = st.selectbox("Selecione um CSV dentro do ZIP:", csv_inside)
                    if selected_csv:
                        with zip_ref.open(selected_csv) as f:
                            stringio = StringIO(f.read().decode('utf-8'))
                            st.session_state.df = pd.read_csv(
                                stringio, low_memory=False, encoding_errors='ignore'
                            )
                            st.session_state.selected_file = selected_csv

        # CSV individual
        elif csv_files:
            uploaded_file = csv_files[0]
            st.session_state.selected_file = uploaded_file.name
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8', errors='ignore'))
            st.session_state.df = pd.read_csv(
                stringio, low_memory=False, encoding_errors='ignore'
            )

        # Excel individual (.xls / .xlsx)
        elif excel_files:
            uploaded_file = excel_files[0]
            st.session_state.selected_file = uploaded_file.name
            st.session_state.df = pd.read_excel(
                uploaded_file,
                engine="openpyxl",
                sheet_name=None  # lê todas as abas
            )
            # Se tiver múltiplas planilhas, une todas
            if isinstance(st.session_state.df, dict):
                combined = []
                for name, df_sheet in st.session_state.df.items():
                    df_sheet['__Planilha__'] = name
                    combined.append(df_sheet)
                st.session_state.df = pd.concat(combined, ignore_index=True)

        # PDFs múltiplos
        elif pdf_files:
            st.info("📂 PDFs carregados — perguntas textuais podem ser feitas ao modelo Gemini (sem dataframe).")
            st.session_state.df = None

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.session_state.df = None

# --- Interação com o Agente Gemini ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_file}' carregado com sucesso.")
    st.dataframe(
        st.session_state.df,
        use_container_width=True,
        height=800  # tabela rolável grande
    )

    st.write(f"**Total de linhas carregadas:** {len(st.session_state.df):,}")

    user_question = st.text_input(
        "Faça uma pergunta sobre os dados:",
        placeholder="Exemplo: Mostre todos os lançamentos de outubro de 2025. Liste como tabela."
    )

    if user_question:
        with st.spinner("O Agente Gemini está analisando todos os registros (sem limitação de linhas)..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )

                AGENT_PREFIX = """
                Você é um Cientista Contábil e Administrador Financeiro com acesso completo aos dados carregados.
                Sua função é analisar todos os registros, **sem limitar linhas** ou filtros automáticos.
                
                🔹 Regras:
                1. Sempre processe o dataset completo.
                2. Para perguntas sobre "tabela" ou "listagem", apresente resultados em formato tabular (pandas DataFrame).
                3. Use lógica contábil e administrativa para interpretar valores, categorias e datas.
                4. Gere gráficos quando for útil (barras, linhas, pizza, heatmap).
                5. Não resuma nem filtre — retorne dados completos.
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

                # Exibir tabela se houver formato tabular
                try:
                    if "|" in output_text and "---" in output_text:
                        table_df = pd.read_csv(StringIO(output_text.replace("|", ",")))
                        st.write("📊 **Tabela Gerada:**")
                        st.dataframe(table_df, use_container_width=True)
                    else:
                        st.success("📈 Resposta do Agente:")
                        st.write(output_text)
                except Exception:
                    st.success("📈 Resposta do Agente:")
                    st.write(output_text)

                fig = plt.gcf()
                if len(fig.get_axes()) > 0:
                    st.write("---")
                    st.subheader("📊 Visualização Gerada")
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
