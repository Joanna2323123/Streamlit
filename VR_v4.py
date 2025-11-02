import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from io import StringIO
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ==============================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ==============================
st.set_page_config(
    page_title="Nexus Quantum | Relatório de Análise de Dados",
    layout="wide",
    page_icon="📊"
)

# ==============================
# ESTILO CUSTOMIZADO (DASHBOARD)
# ==============================
st.markdown("""
<style>
    /* Fundo geral */
    .stApp {
        background: radial-gradient(circle at 25% top, #0f2027, #203a43, #2c5364);
        color: #EAEAEA;
        font-family: 'Inter', sans-serif;
    }
    /* Título principal */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #B5E8FF;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }
    /* Cartões */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        color: #EAEAEA;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }
    .card:hover {
        transform: scale(1.01);
        background: rgba(255,255,255,0.07);
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d4ff;
    }
    /* Caixa lateral */
    .sidebar .sidebar-content {
        background-color: #111927 !important;
    }
    /* Botão */
    button[kind="primary"] {
        background: linear-gradient(90deg, #00b4db, #0083b0);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# CABEÇALHO VISUAL
# ==============================
st.markdown("""
<div style='padding:20px; border-radius:15px; background:linear-gradient(135deg,#0b253a,#092031); margin-bottom:25px;'>
    <h1 class='main-title'>📊 Nexus Quantum | Relatório de Análise Fiscal e Contábil</h1>
    <p class='subtitle'>
        Este painel utiliza IA (Gemini) para gerar insights interativos sobre seus arquivos CSV, ZIP ou PDF. 
        Faça upload dos documentos e explore métricas, tendências e recomendações inteligentes.
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================
# API KEY
# ==============================
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("⚠️ Chave de API do Google não encontrada. Configure-a nos Secrets do Streamlit Cloud.")
    st.stop()

# ==============================
# UPLOAD DE ARQUIVOS
# ==============================
uploaded_files = st.file_uploader(
    "📂 Envie um ou mais arquivos (.zip, .csv ou .pdf)",
    type=["zip", "csv", "pdf"],
    accept_multiple_files=True
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_csv' not in st.session_state:
    st.session_state.selected_csv = ""

# ==============================
# PROCESSAMENTO DE ARQUIVOS
# ==============================
if uploaded_files:
    try:
        pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
        zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
        csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]

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
                            st.session_state.df = pd.read_csv(stringio)
                            st.session_state.selected_csv = selected_csv

        # CSV individual
        elif csv_files:
            uploaded_file = csv_files[0]
            st.session_state.selected_csv = uploaded_file.name
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            st.session_state.df = pd.read_csv(stringio)

        # PDFs múltiplos (sem exibir lista)
        elif pdf_files:
            st.markdown("""
            <div style="padding:20px; border-radius:12px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); text-align:center;">
                <h3 style="color:#00d4ff; margin-bottom:10px;">📂 PDFs carregados com sucesso</h3>
                <p style="opacity:0.8;">Os arquivos foram processados. Agora você pode fazer perguntas textuais ao modelo Gemini.</p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.df = None

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.session_state.df = None

# ==============================
# ANÁLISE DE CSV
# ==============================
if st.session_state.df is not None:
    st.markdown("### 📈 Visualização e Análise")
    st.dataframe(st.session_state.df.head(), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='card'><div class='metric-label'>Linhas</div><div class='metric-value'>" +
                    str(len(st.session_state.df)) + "</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><div class='metric-label'>Colunas</div><div class='metric-value'>" +
                    str(len(st.session_state.df.columns)) + "</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><div class='metric-label'>Arquivo</div><div class='metric-value'>" +
                    st.session_state.selected_csv + "</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='card'><div class='metric-label'>Status</div><div class='metric-value'>✅ Pronto</div></div>", unsafe_allow_html=True)

    user_question = st.text_input(
        "💬 Pergunte algo sobre os dados:",
        placeholder="Exemplo: Qual a correlação entre as variáveis?"
    )

    if user_question:
        with st.spinner("🧠 O Agente Gemini está analisando..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )

                AGENT_PREFIX = """
                Você é um analista de dados experiente. Gere respostas claras, com gráficos e análises visuais.
                1. Se houver menção a "distribuição" ou "variabilidade", gere histograma e boxplot.
                2. Se for "correlação", gere um heatmap.
                3. Se for "valores frequentes", mostre tabelas resumidas (.value_counts()).
                4. Sempre que possível, prefira gráficos a texto.
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
                st.markdown("### 🧾 Resposta da IA")
                st.write(output_text)

                fig = plt.gcf()
                if len(fig.get_axes()) > 0:
                    st.markdown("### 📊 Visualização Gerada")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")

# ==============================
# ANÁLISE DE PDF (TEXTO)
# ==============================
elif uploaded_files and any(f.name.endswith(".pdf") for f in uploaded_files):
    user_question = st.text_input(
        "💬 Pergunte algo sobre o conteúdo dos PDFs:",
        placeholder="Exemplo: Resuma o conteúdo dos documentos enviados."
    )

    if user_question:
        with st.spinner("🧠 O Agente Gemini está lendo os PDFs..."):
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

                response = llm.invoke(f"Baseado neste texto:\n{full_text}\n\nPergunta: {user_question}")
                st.markdown("### 🧾 Resposta da IA")
                st.write(response.content)
            except Exception as e:
                st.error(f"Erro ao processar os PDFs: {e}")
else:
    st.info("⬆️ Envie um arquivo CSV, ZIP ou PDF para iniciar a análise.")
