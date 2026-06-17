import streamlit as st
from utility.session import init_session, show_sidebar
from view import (
    Overview,
    Analisis_Aspek,
    Sentiments_Aspects_Trends,
    Tabel_Data_Komentar,
    Input_Data,
    login,
)

st.set_page_config(
    page_title="ABSA Insights",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Background utama */
        .stApp {
            background-color: #DAE2FD !important;
            color: #000000 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #a5b5d2 !important;
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] * {
            color: #000000 !important;
        }

        /* Input field, textarea, select, dan tombol di dalam input (simbol mata) */
        input, textarea, select, div[data-baseweb="input"] button {
            background-color: #A7C7E7 !important;
            color: #000000 !important;
            border: 0px solid #204d86 !important;
            border-radius: 5px;
        }
        
        /* Tombol */
        .stButton>button {
            background-color: #204d86 !important;
            color: #ffffff !important;
            border-radius: 5px;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #163a63 !important;
            color: #ffffff !important;
        }

        /* Custom Exception Merah Terang di Sidebar */
        section[data-testid="stSidebar"] div[data-testid="stException"] {
            background-color: #b52a2a !important; /* Merah terang solid */
            border: 1px solid #CC0000 !important;
            border-radius: 5px;
        }
        
        /* Memaksa semua teks di dalam kotak exception menjadi putih agar kontras */
        section[data-testid="stSidebar"] div[data-testid="stException"] * {
            color: #FFFFFF !important;
        }

        .altair-chart {
            background-color: transparent !important;
        }

        img {
            max-width: 100% !important;
            height: auto !important;
        }

        header[data-testid="stHeader"] {
            display: block !important;
            background: transparent !important;
            height: 56px !important;
            z-index: 1200 !important;
            box-shadow: none !important;
        }

        .dashboard-header {
            position: fixed;
            top: 0;
            left: 260px;
            width: calc(100% - 260px);
            height: 56px;
            background: #204d86;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.14);
            transition: left 0.2s ease, width 0.2s ease;
        }

        .dashboard-header-title {
            color: #ffffff;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 0;
            margin-left: 0;
        }

        body:has(section[data-testid="stSidebar"][aria-expanded="false"]) .dashboard-header,
        body:not(:has(section[data-testid="stSidebar"])) .dashboard-header {
            left: 0;
            width: 100%;
            padding-left: 64px;
        }

        .block-container {
            padding-top: 86px !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #a5b5d2 0%, #dbe6ff 100%) !important;
            width: 260px !important;
            min-width: 260px !important;
            padding-top: 64px !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }

        .sidebar-menu-anchor,
        .sidebar-logout-anchor {
            display: none;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        section[data-testid="stSidebar"] div:has(> .sidebar-menu-anchor),
        section[data-testid="stSidebar"] div:has(> .sidebar-logout-anchor) {
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            min-height: 42px !important;
            background: rgba(255,255,255,0.38) !important;
            border: 1px solid rgba(32,77,134,0.22) !important;
            color: #0b1c30 !important;
            border-radius: 10px !important;
            justify-content: flex-start !important;
            text-align: left !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease !important;
        }

        section[data-testid="stSidebar"] .stButton > button p {
            color: #0b1c30 !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(32,77,134,0.18) !important;
            border-color: rgba(32,77,134,0.42) !important;
            transform: translateY(-1px);
        }

        section[data-testid="stSidebar"] div:has(> .sidebar-menu-anchor.active) + div .stButton > button,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #204d86 !important;
            border-color: #204d86 !important;
        }

        section[data-testid="stSidebar"] div:has(> .sidebar-menu-anchor.active) + div .stButton > button p,
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] p {
            color: #ffffff !important;
        }

        .menu-desc {
            font-size: 12px;
            margin: 2px 0 8px 0;
            padding: 9px 11px;
            border-radius: 10px;
            background: rgba(255,255,255,0.48);
            color: #1f2937;
            border-left: 3px solid #204d86;
        }

        section[data-testid="stSidebar"] div:has(> .sidebar-logout-anchor) + div .stButton > button {
            background: #b52a2a !important;
            border-color: #b52a2a !important;
            justify-content: center !important;
        }

        section[data-testid="stSidebar"] div:has(> .sidebar-logout-anchor) + div .stButton > button p {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] .st-key-logout_btn button,
        section[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_btn"] button {
            background: #dc2626 !important;
            border-color: #dc2626 !important;
            color: #ffffff !important;
            justify-content: center !important;
        }

        section[data-testid="stSidebar"] .st-key-logout_btn button p,
        section[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_btn"] button p {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] .st-key-logout_btn button:hover,
        section[data-testid="stSidebar"] div[data-testid="stButton"][key="logout_btn"] button:hover {
            background: #b91c1c !important;
            border-color: #b91c1c !important;
        }

        @media (max-width: 1200px) {
            .block-container {
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }

            .dashboard-header {
                left: 240px;
                width: calc(100% - 240px);
            }

            section[data-testid="stSidebar"] {
                width: 240px !important;
                min-width: 240px !important;
            }
        }

        @media (max-width: 900px) {
            .dashboard-header {
                left: 0;
                width: 100%;
                height: 52px;
                padding-left: 58px;
                padding-right: 16px;
            }

            .dashboard-header-title {
                font-size: 20px;
            }

            .block-container {
                padding-top: 72px !important;
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
            }

            div[data-testid="stHorizontalBlock"] {
                gap: 0.75rem !important;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stAltairChart"] {
                max-width: 100% !important;
                overflow-x: auto !important;
            }

            h1 {
                font-size: 1.8rem !important;
                line-height: 1.2 !important;
            }

            h2 {
                font-size: 1.45rem !important;
                line-height: 1.25 !important;
            }

            h3 {
                font-size: 1.1rem !important;
                line-height: 1.25 !important;
            }

            .stButton > button,
            .stDownloadButton > button {
                min-height: 40px !important;
                white-space: normal !important;
            }
        }

        @media (max-width: 640px) {
            .dashboard-header-title {
                font-size: 18px;
            }

            .block-container {
                padding-left: 0.6rem !important;
                padding-right: 0.6rem !important;
            }

            section[data-testid="stSidebar"] {
                width: 82vw !important;
                min-width: 82vw !important;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }

            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 0 !important;
            }

            .stSelectbox,
            .stTextInput,
            .stDateInput,
            .stNumberInput,
            .stTextArea {
                width: 100% !important;
            }
        }
    </style>""",
    unsafe_allow_html=True
)


# Inisialisasi session
init_session()

if not st.session_state.login_status:
    st.session_state.page = "Login"
else:
    st.markdown(
        '<div class="dashboard-header"><div class="dashboard-header-title">ABSA Insight</div></div>',
        unsafe_allow_html=True,
    )


show_sidebar()

# Routing Halaman
if st.session_state.page == "Login":
    login.show()
elif st.session_state.page == "Overview":
    Overview.show()
elif st.session_state.page == "Analisis Aspek":
    Analisis_Aspek.show()
elif st.session_state.page == "Sentiments & Aspects Trends":
    Sentiments_Aspects_Trends.show()
elif st.session_state.page == "Tabel Data Komentar":
    Tabel_Data_Komentar.show()
elif st.session_state.page == "Input Data":
    Input_Data.show()
