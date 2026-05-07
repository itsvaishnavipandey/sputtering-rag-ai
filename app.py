__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import chromadb
import google.generativeai as genai
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Sputtering Intelligence System",
    page_icon="🧪",
    layout="wide"
)

# ==========================================
# FUTURISTIC GLOBAL CSS
# ==========================================

st.markdown("""
<style>

/* =========================
GLOBAL APP
========================= */

.stApp{
    background:
      radial-gradient(circle at top left,
      rgba(0,245,255,0.08), transparent 30%),

      radial-gradient(circle at bottom right,
      rgba(123,97,255,0.08), transparent 30%),

      linear-gradient(180deg,#050816 0%, #081120 100%);

    color:white;
}

/* =========================
REMOVE DEFAULT PADDING
========================= */

.block-container{
    padding-top:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* =========================
SIDEBAR
========================= */

section[data-testid="stSidebar"]{
    background:rgba(8,12,25,0.95);
    border-right:1px solid rgba(255,255,255,0.06);
}

/* =========================
INPUTS
========================= */

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    color:white !important;
    border-radius:12px !important;
}

/* =========================
BUTTONS
========================= */

.stButton>button{
    background:
      linear-gradient(
        90deg,
        rgba(0,245,255,0.15),
        rgba(123,97,255,0.15)
      );

    border:1px solid rgba(0,245,255,0.25);
    border-radius:14px;

    color:#00F5FF;
    font-weight:600;

    transition:0.3s;
}

.stButton>button:hover{
    transform:translateY(-2px);
    border:1px solid rgba(0,245,255,0.5);
    box-shadow:0 0 20px rgba(0,245,255,0.2);
}

/* =========================
METRIC CARDS
========================= */

[data-testid="metric-container"]{
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.06);
    padding:18px;
    border-radius:18px;
    backdrop-filter: blur(12px);
}

/* =========================
TABS
========================= */

.stTabs [data-baseweb="tab-list"]{
    gap:10px;
}

.stTabs [data-baseweb="tab"]{
    background:rgba(255,255,255,0.04);
    border-radius:12px;
    padding:10px 18px;
    color:white;
}

.stTabs [aria-selected="true"]{
    background:linear-gradient(
        90deg,
        rgba(0,245,255,0.18),
        rgba(123,97,255,0.18)
    ) !important;
    border:1px solid rgba(0,245,255,0.25);
}

/* =========================
CHAT
========================= */

.stChatMessage{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:18px;
    padding:14px;
}

/* =========================
DATAFRAME
========================= */

[data-testid="stDataFrame"]{
    border-radius:18px;
    overflow:hidden;
    border:1px solid rgba(255,255,255,0.06);
}

/* =========================
HEADERS
========================= */

.main-title{
    font-size:52px;
    font-weight:800;
    text-align:center;

    background: linear-gradient(
        90deg,
        #00F5FF,
        #7B61FF,
        #FF4FD8
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    margin-bottom:0;
}

.sub-title{
    text-align:center;
    color:#94A3B8;
    font-size:15px;
    margin-top:6px;
    margin-bottom:30px;
}

.glass-panel{
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:22px;
    padding:22px;
    backdrop-filter: blur(14px);
}

/* =========================
SCROLLBAR
========================= */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#1E293B;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="glass-panel">

<h1 class="main-title">
SPUTTERING INTELLIGENCE SYSTEM
</h1>

<p class="sub-title">
AI-Powered Thin Film Deposition Analysis • Material Intelligence • Semantic Research Engine
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_csv_data():
    df = pd.read_csv("./sputtering_database_clean_final.csv")

    numeric_cols = [
        'Power_W',
        'Working_Pressure_Pa',
        'Base_Pressure_Pa',
        'Temperature_C',
        'Thickness_nm'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df = load_csv_data()

@st.cache_resource
def load_database():
    DB_PATH = "./vector_database"
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name="sputtering_papers")

collection = load_database()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ⚙ AI Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    selected_model = st.selectbox(
        "AI Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash"
        ]
    )

    st.markdown("---")

    st.markdown("### 🧠 System Status")

    st.success("Vector Database Online")
    st.success("RAG Pipeline Active")
    st.success("Semantic Retrieval Ready")

    st.markdown("---")

    st.metric("Research Papers", len(df))
    st.metric("Material Families", df['Material'].nunique())

# ==========================================
# MAIN TABS
# ==========================================

tab1, tab2 = st.tabs([
    "📊 Macro Analytics",
    "💬 AI Research Assistant"
])

# ==========================================
# TAB 1
# ==========================================

with tab1:

    st.markdown("## 📈 Semantic Material Analytics")

    material_query = st.text_input(
        "Enter Material Family",
        "ZnO"
    )

    if material_query:

        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
        else:

            with st.spinner(f"Analyzing {material_query} material family..."):

                genai.configure(api_key=api_key)

                search_results = collection.query(
                    query_texts=[material_query],
                    n_results=50
                )

                retrieved_materials = list(set([
                    meta['Material']
                    for meta in search_results['metadatas'][0]
                ]))

                llm = genai.GenerativeModel(selected_model)

                filter_prompt = f"""
                The user is researching the thin film material: '{material_query}'.

                Here is a list of raw material names found in our database:
                {retrieved_materials}

                Which belong to the '{material_query}' family?

                Return ONLY comma-separated exact strings.
                """

                try:

                    valid_materials_text = llm.generate_content(
                        filter_prompt
                    ).text

                    valid_materials = [
                        m.strip()
                        for m in valid_materials_text.split(',')
                    ]

                    st.info(
                        f"AI grouped these materials: {', '.join(valid_materials)}"
                    )

                    filtered_df = df[
                        df['Material'].isin(valid_materials)
                    ].copy()

                    filtered_df['Substrate'] = (
                        filtered_df['Substrate']
                        .astype(str)
                        .str.title()
                        .str.strip()
                    )

                    st.write(
                        f"### 🔬 {len(filtered_df)} Papers Found"
                    )

                    if len(filtered_df) > 0:

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "⚡ Avg Power",
                                f"{filtered_df['Power_W'].mean():.1f} W"
                            )

                        with col2:
                            st.metric(
                                "🌡 Avg Temp",
                                f"{filtered_df['Temperature_C'].mean():.1f} °C"
                            )

                        with col3:
                            st.metric(
                                "💨 Avg Pressure",
                                f"{filtered_df['Working_Pressure_Pa'].mean():.4f} Pa"
                            )

                        with col4:
                            st.metric(
                                "📚 Papers",
                                len(filtered_df)
                            )

                        st.markdown("---")

                        row1_col1, row1_col2 = st.columns(2)

                        with row1_col1:

                            fig_pow = px.histogram(
                                filtered_df,
                                x="Power_W",
                                nbins=40,
                                title="Target Power Distribution",
                                color_discrete_sequence=['#00F5FF']
                            )

                            fig_pow.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )

                            st.plotly_chart(
                                fig_pow,
                                use_container_width=True
                            )

                        with row1_col2:

                            fig_temp = px.histogram(
                                filtered_df,
                                x="Temperature_C",
                                nbins=40,
                                title="Temperature Distribution",
                                color_discrete_sequence=['#7B61FF']
                            )

                            fig_temp.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )

                            st.plotly_chart(
                                fig_temp,
                                use_container_width=True
                            )

                        fig_press = px.histogram(
                            filtered_df,
                            x="Working_Pressure_Pa",
                            nbins=40,
                            title="Working Pressure Distribution",
                            color_discrete_sequence=['#FF4FD8']
                        )

                        fig_press.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )

                        st.plotly_chart(
                            fig_press,
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"API Error: {e}")

# ==========================================
# TAB 2
# ==========================================

with tab2:

    st.markdown("## 🤖 AI Research Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "Ask about sputtering parameters, materials, substrates..."
    ):

        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        if not api_key:
            st.error("Please enter your Gemini API Key.")
            st.stop()

        genai.configure(api_key=api_key)

        llm = genai.GenerativeModel(selected_model)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching scientific database..."
            ):

                results = collection.query(
                    query_texts=[prompt],
                    n_results=10
                )

                retrieved_docs = results['documents'][0]

                context = "\n\n".join(retrieved_docs)

                system_prompt = f"""
                You are an expert materials science AI assistant.

                Use ONLY the provided database context.

                Context:
                {context}

                User Question:
                {prompt}
                """

                try:

                    response = llm.generate_content(system_prompt)

                    ai_reply = response.text

                    sources_text = "\n\n### 📚 Sources\n"

                    for meta in results['metadatas'][0]:

                        sources_text += (
                            f"- {meta['Paper_ID']} "
                            f"({meta['Material']})\n"
                        )

                    full_response = ai_reply + sources_text

                    st.markdown(full_response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                except Exception as e:
                    st.error(f"API Error: {e}")
