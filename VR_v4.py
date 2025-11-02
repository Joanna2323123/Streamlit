import streamlit as st
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
from io import StringIO

# Importações necessárias do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Configuração da Página (Atualizada) ---
st.set_page_config(
    page_title="Analisador Fiscal (NEXUS)", # MUDADO
    page_icon="🧾", # MUDADO
    layout="wide"
)
st.title("🧾 Análise de Dados Fiscais com Agente Gemini") # MUDADO
st.write(
    "Faça o upload de um arquivo `.zip` contendo um ou mais CSVs. "
    "O agente usará o modelo Gemini do Google para responder perguntas sobre seus dados e gerar visualizações."
)

# --- Configuração da Chave de API (Sem alteração) ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não encontrada. Por favor, configure-a nos 'Secrets' do seu aplicativo no Streamlit Cloud.")
    st.stop()


# --- Lógica de Upload e Seleção de Arquivo (Movido para a Barra Lateral) ---
with st.sidebar:
    st.header("Configuração de Upload")
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo .zip",
        type="zip"
    )

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
                    st.session_state.selected_csv = selected_csv
                    with zip_ref.open(selected_csv) as f:
                        stringio = StringIO(f.read().decode('utf-8'))
                        # Tenta ler com 'latin1' se 'utf-8' falhar
                        try:
                            st.session_state.df = pd.read_csv(stringio)
                        except UnicodeDecodeError:
                            stringio.seek(0)
                            st.session_state.df = pd.read_csv(stringio, encoding='latin1')

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.session_state.df = None
else:
    st.info("Aguardando o upload de um arquivo .zip na barra lateral para iniciar a análise.")
    st.stop() # Para a execução se nenhum arquivo for carregado

# --- Interação com o Agente ---
if st.session_state.df is not None:
    st.success(f"Arquivo '{st.session_state.selected_csv}' carregado.")
    
    # --- 1. NOVA SEÇÃO: BALANCETE CONTÁBIL (DASHBOARD) ---
    st.subheader("📊 Balancete Contábil (Visão Geral)")
    
    df = st.session_state.df
    
    # --- Mapeamento de Colunas (AJUSTE CONFORME SEU CSV) ---
    # Tenta adivinhar nomes de colunas comuns para dados fiscais.
    # Se os nomes no seu CSV forem diferentes, ajuste-os aqui.
    COL_MAP = {
        "total_nfe": "vNF",         # Nomes comuns: "valor_total_nfe", "valor_nota", "vNF"
        "total_produtos": "vProd",  # Nomes comuns: "valor_total_produtos", "vProd"
        "icms": "vICMS",            # Nomes comuns: "valor_icms", "vICMS"
        "pis": "vPIS",              # Nomes comuns: "valor_pis", "vPIS"
        "cofins": "vCOFINS",        # Nomes comuns: "valor_cofins", "vCOFINS"
        "iss": "vISSQN",            # Nomes comuns: "valor_iss", "vISSQN"
        "estado": "UF"              # Nomes comuns: "dest_uf", "estado_dest", "UF"
    }

    # Função auxiliar para exibir as métricas como nas imagens
    def show_metric(df, col_key, title):
        col_name = COL_MAP.get(col_key)
        value = "R$ 0,00"
        help_text = f"Coluna '{col_name}' não encontrada. Verifique os nomes no seu CSV e ajuste o 'COL_MAP' no código."
        delta_text = "Atenção"
        
        if col_name and col_name in df.columns:
            try:
                # Tenta converter a coluna para número, tratando erros (ex: "1.234,56")
                numeric_col = pd.to_numeric(df[col_name].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
                
                if numeric_col.isnull().all():
                    help_text = f"Coluna '{col_name}' encontrada, mas todos os valores estão nulos ou não são numéricos."
                else:
                    total = numeric_col.sum()
                    value = f"R$ {total:,.2f}"
                    help_text = f"Soma total da coluna '{col_name}'."
                    delta_text = "Calculado"
            except Exception as e:
                value = "Erro"
                help_text = f"Erro ao calcular '{col_name}': {e}"
                delta_text = "Erro"

        st.metric(label=title, value=value, help=help_text, 
                  delta=delta_text if delta_text != "Calculado" else None, 
                  delta_color="inverse" if delta_text != "Calculado" else "normal")

    # Exibe as métricas principais em colunas
    col1, col2, col3 = st.columns(3)
    with col1:
        total_itens = len(df)
        st.metric(label="Total de Itens Processados", value=total_itens, help="Número total de linhas (itens) no CSV.")
        show_metric(df, "icms", "Valor Total de ICMS")
        
    with col2:
        show_metric(df, "total_nfe", "Valor Total das NFes")
        show_metric(df, "pis", "Valor Total de PIS")
        
    with col3:
        show_metric(df, "total_produtos", "Valor Total dos Produtos")
        show_metric(df, "cofins", "Valor Total de COFINS")

    # Métrica de ISS (separada)
    show_metric(df, "iss", "Valor Total de ISS")
    st.markdown("---") # Separador

    # --- 2. NOVA SEÇÃO: ICMS POR ESTADO ---
    with st.expander("🧾 Análise de ICMS por Estado"):
        icms_col = COL_MAP.get("icms")
        estado_col = COL_MAP.get("estado")
        
        if icms_col and estado_col and icms_col in df.columns and estado_col in df.columns:
            try:
                df[icms_col] = pd.to_numeric(df[icms_col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
                
                # Agrupa, soma e ordena
                icms_por_estado = df.groupby(estado_col)[icms_col].sum().sort_values(ascending=False)
                
                st.dataframe(icms_por_estado.map("R$ {:,.2f}".format))
                
                # Gráfico de barras
                st.bar_chart(icms_por_estado)
                
            except Exception as e:
                st.warning(f"Não foi possível calcular o ICMS por estado. Colunas encontradas, mas ocorreu um erro: {e}")
        else:
            st.info(f"Para ver o ICMS por estado, o arquivo CSV precisa ter as colunas '{icms_col}' e '{estado_col}'. (Nomes de colunas definidos no 'COL_MAP')")
            
    st.markdown("---")
    
    # --- 3. SEÇÃO EXISTENTE: CHAT COM O AGENTE (COM PREFIXO CORRIGIDO) ---
    st.subheader("💬 Chat Interativo com Agente")
    
    # O placeholder foi atualizado para refletir o foco fiscal
    user_question = st.text_input(
        "❓ Faça uma pergunta sobre os dados:",
        placeholder="Qual o cliente com maior valor? Qual o faturamento total?" 
    )
    
    if user_question:
        with st.spinner("O Agente Gemini está pensando..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=google_api_key
                )
                
                # --- 4. CORREÇÃO CRÍTICA: AGENT_PREFIX ---
                # O seu prefixo antigo era sobre estatística.
                # Este prefixo é focado em Análise Fiscal (NEXUS) e lida com erros.
                AGENT_PREFIX = """
                Você é o "NEXUS", um agente especialista em análise de dados Fiscais e Financeiros. Seja direto, mas também robusto em suas respostas.

                **SUAS REGRAS DE COMPORTAMENTO:**

                1.  **VERIFICAÇÃO DE COLUNAS (REGRA MAIS IMPORTANTE):**
                    * **ANTES** de tentar responder a uma pergunta que exige colunas específicas (como 'ICMS', 'PIS', 'COFINS', 'cliente', 'vNF', 'natureza_da_operação'), **PRIMEIRO** verifique se essas colunas existem em `df.columns`.
                    * Se as colunas **NÃO EXISTIREM**, sua resposta **DEVE** ser informar ao usuário quais colunas estão faltando para aquela análise.
                    * **Exemplo de Resposta de Falha:** "Não posso calcular. As colunas 'ICMS', 'PIS' e 'vNF' não foram encontradas nos dados."
                    * **NÃO FALHE EM SILÊNCIO.**

                2.  **PERGUNTAS GENÉRICAS (MÉTRICAS NEXUS):**
                    * Se o usuário fizer uma pergunta genérica ("Quais os principais dados?", "resumo", "métricas", "insights") E as colunas necessárias existirem, calcule as métricas de negócio principais (Faturamento Total, Cliente de Maior Valor, Ticket Médio).
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
                output_text = response.get("output", "Não foi possível gerar umaVAI.")

                st.success("Resposta do Agente:")
                st.write(output_text)
                
                fig = plt.gcf()
                if len(fig.get_axes()) > 0:
                    st.write("---")
                    st.subheader("📊 Gráfico Gerado")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do agente: {e}")
