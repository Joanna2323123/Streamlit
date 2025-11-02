import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
from PyPDF2 import PdfReader
# --- LangChain / Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Nexus QuantumAI - Análise Fiscal e Contábil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adiciona um estilo customizado para o tema escuro e visual mais limpo (CSS)
st.markdown("""
    <style>
        /* Fundo principal escuro */
        .stApp {
            background-color: #0d1117;
            color: #ffffff;
        }
        /* Cor dos cabeçalhos */
        h1, h2, h3, h4, .st-b5 {
            color: #00c7a8; /* Um verde/azul neon */
        }
        /* Estilo para caixas de insights (similar ao visual do painel) */
        .insight-box {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 5px solid #00c7a8;
            background-color: #161b22;
        }
        /* Métrica com destaque */
        .stMetric label {
            font-size: 14px;
            color: #9c9d9f;
        }
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 24px;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)


# --- 2. Variáveis de Estado ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_csv' not in st.session_state:
    st.session_state.selected_csv = ""

# --- 3. Chave de API ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Configure-a nos 'Secrets' do seu aplicativo Streamlit Cloud.")
    st.stop()


# --- 4. Sidebar para Upload de Arquivos ---
with st.sidebar:
    st.header("Upload de Arquivos")
    uploaded_files = st.file_uploader(
        "Envie um ou mais arquivos (.zip, .csv ou .pdf)",
        type=["zip", "csv", "pdf"],
        accept_multiple_files=True
    )
    st.markdown("---")
    st.info("O Agente Gemini usa o modelo `gemini-2.5-flash` para analisar seus dados e gerar gráficos.")


# --- 5. Funções de Processamento de Arquivos ---
def process_uploaded_files(uploaded_files):
    pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
    zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
    csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]
    
    # Prioridade para CSV/ZIP para análise de DataFrame
    if zip_files:
        uploaded_file = zip_files[0]
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            csv_inside = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_inside:
                selected_csv = st.selectbox("Selecione um CSV dentro do ZIP:", csv_inside, key="zip_select")
                if selected_csv:
                    with zip_ref.open(selected_csv) as f:
                        # Tenta ler com utf-8, se falhar, tenta latin1 para compatibilidade
                        try:
                            stringio = StringIO(f.read().decode('utf-8'))
                            st.session_state.df = pd.read_csv(stringio)
                        except UnicodeDecodeError:
                            f.seek(0)
                            stringio = StringIO(f.read().decode('latin1'))
                            st.session_state.df = pd.read_csv(stringio)
                        st.session_state.selected_csv = selected_csv
    
    elif csv_files:
        uploaded_file = csv_files[0]
        st.session_state.selected_csv = uploaded_file.name
        # Tenta ler com utf-8, se falhar, tenta latin1
        try:
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            st.session_state.df = pd.read_csv(stringio)
        except UnicodeDecodeError:
            stringio = StringIO(uploaded_file.getvalue().decode('latin1'))
            st.session_state.df = pd.read_csv(stringio)


# --- 6. Layout Principal e Lógica de Análise ---
if uploaded_files:
    try:
        process_uploaded_files(uploaded_files)
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.session_state.df = None

# --- 7. Dashboard de Análise de Dados (Se houver DataFrame) ---
if st.session_state.df is not None:
    df = st.session_state.df
    st.header("Relatório de Análise Fiscal e Contábil")
    st.markdown(f"Este relatório apresenta uma análise resumida de **{len(df)}** registros do arquivo **{st.session_state.selected_csv}**.")
    st.markdown("---")
    
    # 7.1. Colunas Principais: Conteúdo (2/3) e Chat (1/3)
    main_content_col, chat_col = st.columns([2, 1])

    with main_content_col:
        st.subheader("📊 Métricas Chave")
        
        # 7.1.1. KPIs DINÂMICOS
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        total_docs = len(df)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # --- LÓGICA DINÂMICA PARA VALORES ---
        valor_total_nfe = "N/A"
        icms_index = "N/A"
        value_col_name = "Valor (Números)" # Nome padrão

        if len(numeric_cols) > 0:
            # Usa a soma da primeira coluna numérica como 'Valor Total'
            value_col_name = numeric_cols[0] 
            total_value = df[value_col_name].sum()
            # Formatação para moeda BR
            valor_total_nfe = f"R$ {total_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            # KPI de exemplo (Índice de Conformidade) - SIMULADO
            icms_compliance_rate = 0.95 
            icms_index = f"{icms_compliance_rate * 100:.1f}%"
            
        # Simulação de Risco (Dinâmico, baseado na quantidade de dados)
        risco_tributario = "Médio" if total_docs > 5000 and len(numeric_cols) > 0 else "Baixo"
        # ------------------------------------

        with kpi_col1:
            st.metric("Documentos Válidos", total_docs)
            
        with kpi_col2:
            st.metric(f"Valor Total ({value_col_name})", valor_total_nfe)

        with kpi_col3:
            st.metric("Índice Conformidade ICMS (Sim.)", icms_index, delta="0.5%", delta_color="normal")
        
        with kpi_col4:
            # Destaque de Risco
            color = "orange" if risco_tributario == "Médio" else "green"
            st.markdown("Nível Risco Tributário (Sim.)")
            st.markdown(f'<p style="color: {color}; font-size: 24px; font-weight: bold;">{risco_tributario}</p>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 7.1.2. Gráfico de Tendência DINÂMICO e AGREGADO (Mensal)
        st.subheader("📈 Tendência do Valor Total (Mensal)")

        df_to_plot = df.copy()
        
        # 1. Tenta identificar Coluna de Data/Tempo
        date_cols = [c for c in df_to_plot.columns if any(keyword in c.lower() for keyword in ['data', 'emissao', 'mes', 'dt'])]
        
        # 2. Tenta identificar Coluna de Valor
        value_cols = [c for c in df_to_plot.columns if any(keyword in c.lower() for keyword in ['valor', 'total', 'preco'])]
        
        # 3. Lógica de Plotagem
        if date_cols and value_cols:
            date_col = date_cols[0]
            value_col = value_cols[0]
            
            try:
                # Conversão robusta da coluna de data
                df_to_plot[date_col] = pd.to_datetime(df_to_plot[date_col], errors='coerce', dayfirst=True)
                df_to_plot.dropna(subset=[date_col, value_col], inplace=True)
                
                # Agregação Mensal
                df_to_plot['Período'] = df_to_plot[date_col].dt.to_period('M')
                
                df_mensal = df_to_plot.groupby('Período')[value_col].sum().reset_index()
                df_mensal['Período'] = df_mensal['Período'].astype(str) # Converte para string para Plotly
                
                # Gera o Gráfico
                fig = px.line(df_mensal, x='Período', y=value_col, 
                              title=f'Soma Mensal de "{value_col}"',
                              template='plotly_dark',
                              labels={'Período': 'Mês de Referência', value_col: f'Soma de {value_col}'})
                
                # Adiciona formatação para o eixo Y
                fig.update_yaxes(tickprefix="R$ ", separatethousands=True)
                
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Erro ao processar e plotar os dados mensais: Verifique se '{date_col}' e '{value_col}' têm formatos válidos. Erro: {e}")
                st.info("Para gráficos mais complexos, use o **Chat Interativo com IA**.")
                
        else:
            st.info("Não foi possível gerar um gráfico de tendência automático. Requisito: Coluna de Data/Emissão e Coluna de Valor/Total.")


        # 7.1.3. Insights Acionáveis (Exemplo)
        st.subheader("💡 Insights Acionáveis (Exemplo)")
        st.markdown("""
            <div class="insight-box">
                * **Priorizar** a revisão das operações interestaduais para assegurar o correto recolhimento do **DIFAL**.
                * Auditar as Notas Fiscais com **"NATUREZA DE OPERAÇÃO"** de **'REMESSA'** ou **'RETORNO'** para conformidade.
                * Implementar um sistema de conciliação automática para corrigir inconsistências.
                * O **Nível de Risco** requer validação periódica das regras fiscais.
            </div>
        """, unsafe_allow_html=True)

    with chat_col:
        # 7.2. Chat Interativo com IA (Mantido Dinâmico)
        st.subheader("🤖 Chat Interativo com IA")
        st.info("Olá! Sou seu Agente AI. Use a caixa de texto abaixo para fazer perguntas sobre o DataFrame carregado.")
        
        user_question = st.text_input(
            "Pergunte sobre os dados:",
            placeholder="Exemplo: Qual a média da coluna 'Valor Total'?"
        )

        if user_question:
            with st.spinner("O Agente Gemini está analisando..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0,
                        google_api_key=google_api_key
                    )

                    AGENT_PREFIX = """
                    Você é um agente especialista em análise de dados. Sua principal função é fornecer insights através de texto e, se solicitado, visualizações.
                    Regras:
                    1. Use a ferramenta `python_repl_ast` para analisar o DataFrame.
                    2. Para perguntas sobre "distribuição" ou "variância", gere um histograma ou boxplot usando `matplotlib.pyplot`.
                    3. Se a pergunta envolver a relação entre duas variáveis, considere um gráfico de dispersão ou um `heatmap` de correlação.
                    4. O DataFrame está carregado na variável `df`.
                    """

                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
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

                    # Verifica se um gráfico foi gerado pelo agente
                    fig = plt.gcf()
                    if len(fig.get_axes()) > 0:
                        st.subheader("Gráfico Gerado pelo Agente")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a execução do agente: {e}")

# --- 8. Seção para Análise de PDFs (Texto) ---
elif uploaded_files and any(f.name.endswith(".pdf") for f in uploaded_files):
    st.header("Análise de Documentos (PDF)")
    st.markdown("Você carregou documentos PDF. Use o chat para perguntas sobre o texto.")
    
    user_question = st.text_input(
        "Pergunte algo sobre o texto dos PDFs:",
        placeholder="Exemplo: Resuma as principais conclusões do documento."
    )
    if user_question:
        with st.spinner("O Agente Gemini está analisando o PDF..."):
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
                st.success("Resposta do Agente:")
                st.write(response.content)
            except Exception as e:
                st.error(f"Erro ao processar o PDF: {e}")

# --- 9. Mensagem Inicial ---
else:
    st.markdown("""
        <div style="padding: 20px; border: 1px solid #00c7a8; border-radius: 5px; text-align: center;">
            <h2 style="color: #00c7a8;">Bem-vindo ao Nexus QuantumAI</h2>
            <p>Faça o upload de seus arquivos (CSV, ZIP ou PDF) no painel lateral para iniciar a análise e obter insights acionáveis com o Agente Gemini.</p>
        </div>
    """, unsafe_allow_html=True)
