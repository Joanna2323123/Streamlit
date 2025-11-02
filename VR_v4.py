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
    page_title="Nexus QuantumAI - Análise Fiscal",
    layout="wide",
    initial_sidebar_state="collapsed" # Colapsa a sidebar de upload
)

# --- 2. CSS CUSTOMIZADO (Para o visual "Nexus QuantumAI") ---
st.markdown("""
    <style>
        /* Fundo principal e cor do texto */
        .stApp {
            background-color: #0d1117; /* Fundo escuro */
            color: #c9d1d9; /* Cor do texto principal */
        }
        
        /* Cor de destaque para títulos */
        h1, h2, h3, h4, .st-b5 {
            color: #00c7a8; /* Verde-água/Teal */
        }

        /* Estilo dos Cards (para insights) */
        .card {
            background-color: #161b22; /* Fundo do card */
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
        }
        
        /* Estilo dos Insights Acionáveis */
        .insight-list li {
            color: #c9d1d9;
            margin-bottom: 8px;
        }
        .insight-list li::before {
            content: '🔹'; /* Marcador personalizado */
            color: #00c7a8; /* Cor do marcador */
            font-weight: bold;
            display: inline-block;
            width: 1em;
            margin-left: -1em;
        }
        
        /* Estilo das Métricas (KPIs) */
        .stMetric {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 14px;
        }
        .stMetric label { /* "Documentos Válidos", etc. */
            font-size: 14px;
            color: #8b949e; /* Cinza claro */
        }
        .stMetric div[data-testid="stMetricValue"] { /* O valor */
            font-size: 24px;
            font-weight: 600;
            color: #c9d1d9; /* Branco suave */
        }
        
        /* Estilo para Risco Tributário (Markdown) */
        .risk-label {
            font-size: 14px;
            color: #8b949e;
        }
        .risk-value-medium {
            font-size: 24px;
            font-weight: 600;
            color: #f85149; /* Vermelho para Risco Médio/Alto */
        }
        .risk-value-low {
            font-size: 24px;
            font-weight: 600;
            color: #3fb950; /* Verde para Risco Baixo */
        }
        
        /* Estilo da bolha de chat */
        .chat-bubble {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 20px;
            color: #c9d1d9;
        }
        
        /* Oculta o menu "Made with Streamlit" */
        footer {
            visibility: hidden;
        }
        
        /* Estilo da barra de upload (sidebar) */
        .css-1d391kg {
            background-color: #161b22;
        }

    </style>
    """, unsafe_allow_html=True)


# --- 3. Variáveis de Estado ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'selected_csv' not in st.session_state:
    st.session_state.selected_csv = ""

# --- 4. Chave de API ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Configure-a nos 'Secrets'.")
    st.stop()


# --- 5. Sidebar para Upload de Arquivos ---
with st.sidebar:
    st.header("Upload de Arquivos")
    uploaded_files = st.file_uploader(
        "Envie um ou mais arquivos (.zip, .csv ou .pdf)",
        type=["zip", "csv", "pdf"],
        accept_multiple_files=True
    )
    st.markdown("---")
    st.info("O Agente Gemini usa o modelo `gemini-2.5-flash` para analisar seus dados.")

# --- 6. Funções de Processamento de Arquivos ---
def process_uploaded_files(uploaded_files):
    pdf_files = [f for f in uploaded_files if f.name.endswith(".pdf")]
    zip_files = [f for f in uploaded_files if f.name.endswith(".zip")]
    csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]
    
    if zip_files:
        uploaded_file = zip_files[0]
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            csv_inside = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_inside:
                # Seleção de CSV removida para simplicidade, pega o primeiro
                selected_csv = csv_inside[0]
                with zip_ref.open(selected_csv) as f:
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
        try:
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            st.session_state.df = pd.read_csv(stringio)
        except UnicodeDecodeError:
            stringio = StringIO(uploaded_file.getvalue().decode('latin1'))
            st.session_state.df = pd.read_csv(stringio)

# Processa os arquivos imediatamente
if uploaded_files:
    try:
        process_uploaded_files(uploaded_files)
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.session_state.df = None


# --- 7. Título Principal e Abas de Navegação ---
st.title("Nexus QuantumAI")
st.markdown("Interactive Insight & Intelligence from Fiscal Analysis")

tab1, tab2, tab3 = st.tabs(["Análise Executiva", "Simulador Tributário", "Análise Comparativa"])

# --- 8. Conteúdo da Aba "Análise Executiva" ---
with tab1:
    
    # --- 8.1. Layout Principal (Conteúdo | Chat) ---
    main_col, chat_col = st.columns([2, 1])

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # --- 8.1.1. COLUNA PRINCIPAL (ESQUERDA) ---
        with main_col:
            st.markdown("### Relatório de Análise Fiscal e Contábil")
            st.markdown(f"Este relatório apresenta uma análise resumida de **{len(df)}** registros do arquivo **{st.session_state.selected_csv}**.")
            st.markdown("---")
            
            st.markdown("#### Métricas Chave")
            
            # --- KPIs DINÂMICOS ---
            total_docs = len(df)
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            valor_total_nfe = "N/A"
            icms_index = "95,0%" # Simulado (Substitua pela sua lógica)
            value_col_name = "Valor (Números)"

            if len(numeric_cols) > 0:
                # Tenta encontrar colunas de valor
                value_cols = [c for c in numeric_cols if any(keyword in c.lower() for keyword in ['valor', 'total', 'preco'])]
                if value_cols:
                    value_col_name = value_cols[0]
                    total_value = df[value_col_name].sum()
                else:
                    value_col_name = numeric_cols[0]
                    total_value = df[value_col_name].sum()
                
                valor_total_nfe = f"R$ {total_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            risco_tributario = "Médio" if total_docs > 5000 else "Baixo"
            risco_color_class = "risk-value-medium" if risco_tributario == "Médio" else "risk-value-low"
            # --------------------
            
            # LINHA 1 DE KPIs
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.metric("Documentos Válidos", total_docs)
            with kpi_col2:
                st.metric(f"Valor Total NF-e ({value_col_name})", valor_total_nfe)
            with kpi_col3:
                st.metric("Valor Total Produto", valor_total_nfe) # Simulado (use sua lógica)

            # LINHA 2 DE KPIs
            kpi_col4, kpi_col5, kpi_col6 = st.columns(3)
            with kpi_col4:
                st.metric("Índice Conformidade ICMS", icms_index, delta="0.5%")
            with kpi_col5:
                # KPI de Risco (Estilizado com Markdown)
                st.markdown(f"""
                    <div class="risk-label">Nível Risco Tributário</div>
                    <div class="{risco_color_class}">{risco_tributario}</div>
                """, unsafe_allow_html=True)
            with kpi_col6:
                st.metric("Estimativa NVA", "R$ 0,00") # Simulado
            
            st.markdown("---")

            # --- Gráfico de Tendência (Dinâmico) ---
            st.markdown("#### Tendência do Valor Total das NFes")
            
            date_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['data', 'emissao', 'mes', 'dt'])]
            
            if date_cols and value_cols:
                date_col = date_cols[0]
                value_col = value_cols[0]
                
                try:
                    df_to_plot = df.copy()
                    df_to_plot[date_col] = pd.to_datetime(df_to_plot[date_col], errors='coerce', dayfirst=True)
                    df_to_plot.dropna(subset=[date_col, value_col], inplace=True)
                    
                    df_to_plot['Período'] = df_to_plot[date_col].dt.to_period('M')
                    df_mensal = df_to_plot.groupby('Período')[value_col].sum().reset_index()
                    df_mensal['Período'] = df_mensal['Período'].astype(str)
                    
                    fig = px.line(df_mensal, x='Período', y=value_col, 
                                  title=f'Soma Mensal de "{value_col}"',
                                  template='plotly_dark')
                    
                    fig.update_traces(line_color='#00c7a8', line_width=2) # Cor da linha
                    fig.update_layout(
                        plot_bgcolor='#161b22', # Fundo do gráfico
                        paper_bgcolor='#161b22', # Fundo do papel
                        font_color='#c9d1d9',
                        xaxis_gridcolor='#30363d',
                        yaxis_gridcolor='#30363d'
                    )
                    fig.update_yaxes(tickprefix="R$ ", separatethousands=True)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Erro ao gerar gráfico de tendência: {e}")
            else:
                st.info("Não foi possível gerar gráfico de tendência (colunas de Data/Valor não encontradas).")
            
            st.markdown("---")

            # --- Insights (Baseado na Imagem 1) ---
            st.markdown("#### Insights Acionáveis")
            st.markdown("""
                <div class="card">
                    <ul class="insight-list">
                        <li>Priorizar a revisão das operações interestaduais destinadas a não contribuintes de ICMS para assegurar a correta apuração e recolhimento do Diferencial de Alíquotas (DIFAL), mitigando riscos fiscais e potenciais multas.</li>
                        <li>Auditar as Notas Fiscais com 'NATUREZA DA OPERAÇÃO' de 'REMESSA - ENTREGA FUTURA' e 'RETORNO DE MATERIAL' para verificar a existência e a conformidade dos documentos fiscais de acompanhamento e encerramento, como notas de faturamento ou novas remessas.</li>
                        <li>Implementar um sistema de conciliação automática entre os valores de cabeçalho e de itens das NF-e para identificar e corrigir inconsistências decorrentes de truncamento de dados ou erros de digitação, melhorando a acurácia contábil.</li>
                        <li>O 'Índice de Conformidade de ICMS' elevado sugere processos fiscais robustos, mas a monitoria contínua é crucial.</li>
                        <li>Um 'Nível de Risco Tributário' baixo é positivo, mas requer validação periódica das regras fiscais para se manter.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # --- 8.1.2. COLUNA DE CHAT (DIREITA) ---
        with chat_col:
            st.markdown("#### Chat Interativo com IA")
            st.markdown("""
                <div class="chat-bubble">
                Olá! Sou o Nexus AI. Analisei seu relatório e estou pronto para ajudar. O que você gostaria de saber?
                </div>
            """, unsafe_allow_html=True)
            
            user_question = st.text_input(
                "Pergunte sobre o relatório...",
                placeholder="Ex: Qual o valor total por CFOP?"
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
                        3. O DataFrame está carregado na variável `df`.
                        4. Responda em Português.
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

                        fig = plt.gcf()
                        if len(fig.get_axes()) > 0:
                            st.subheader("Gráfico Gerado pelo Agente")
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a execução do agente: {e}")

    # --- 8.2. Seção para Análise de PDFs (se nenhum CSV/ZIP for carregado) ---
    elif uploaded_files and any(f.name.endswith(".pdf") for f in uploaded_files):
        with main_col:
            st.error("A Análise Executiva de Métricas requer um arquivo .csv ou .zip.")
        with chat_col:
            st.markdown("#### Chat Interativo com IA (Modo PDF)")
            user_question = st.text_input(
                "Pergunte algo sobre o texto dos PDFs:",
                placeholder="Ex: Resuma o conteúdo do PDF."
            )
            if user_question:
                with st.spinner("O Agente Gemini está analisando o PDF..."):
                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=google_api_key)
                        full_text = ""
                        for pdf in [f for f in uploaded_files if f.name.endswith(".pdf")]:
                            reader = PdfReader(pdf)
                            for page in reader.pages:
                                full_text += page.extract_text() or ""

                        response = llm.invoke(f"Responda em Português com base neste texto:\n{full_text}\n\nPergunta: {user_question}")
                        st.success("Resposta do Agente:")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Erro ao processar o PDF: {e}")

    # --- 8.3. Tela Inicial (Nenhum arquivo) ---
    else:
        st.markdown(
            """
            <div class="card" style="text-align: center; padding: 40px;">
                <h2 style="color: #00c7a8;">Bem-vindo ao Nexus QuantumAI</h2>
                <p style="font-size: 16px; color: #8b949e;">Faça o upload de seus arquivos (CSV, ZIP ou PDF) na barra lateral esquerda para iniciar a análise e obter insights acionáveis com o Agente Gemini.</p>
                <p style="font-size: 16px; color: #8b949e;">(Clique no <strong>></strong> no canto superior esquerdo para abrir a barra de upload)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# --- 9. Conteúdo das Outras Abas (Placeholders) ---
with tab2:
    st.header("Simulador Tributário")
    st.info("Funcionalidade em desenvolvimento.")

with tab3:
    st.header("Análise Comparativa")
    st.info("Funcionalidade em desenvolvimento.")

