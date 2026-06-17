import streamlit as st
from pathlib import Path


USERNAME = "admin"
PASSWORD = "admin123"

# Path untuk logo
DASHBOARD_DIR = Path(__file__).resolve().parents[1]
LOGO_TULISAN_PATH = DASHBOARD_DIR / "Asset" / "Logo tulisan (remove bg).png"


def show():
    # Center logo
    col1, col2, col3 = st.columns([1.5,2,1])
    with col2:
        if LOGO_TULISAN_PATH.exists():
            st.image(str(LOGO_TULISAN_PATH), width=500)
    
    st.write("")
    st.title("Login ABSA Insight")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login", width='stretch'):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.login_status = True
            st.session_state.page = "Overview"
            st.success("✅ Login berhasil")
            st.rerun()
        else:
            st.error("❌ Username atau password salah")
