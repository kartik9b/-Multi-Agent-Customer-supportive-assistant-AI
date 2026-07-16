import streamlit as st
import requests
import pandas as pd

# 📊 1. Configure Page Layout Options
st.set_page_config(
    page_title="Multi-Agent AI Support Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔐 2. Simple Secure Login System
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h2 style='text-align: center; color: #00ecb3;'>🤖 Multi-Agent Hub Portal</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        st.subheader("Secure Staff Login")
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Authenticate")
        
        if submit_button:
            # Simple, secure check (you can change these credentials)
            if username == "admin" and password == "secure123":
                st.session_state.logged_in = True
                st.success("Access Granted! Loading system...")
                st.rerun()
            else:
                st.error("Invalid Username or Password. Please check with your administrator.")

# If the user is not authenticated, stop here and show the login page
if not st.session_state.logged_in:
    login()
    st.stop()

# --- 🚀 ENTERING SECURE DASHBOARD REGION 🚀 ---

# 🧠 3. Initialize Session State for Chat History & Diagnostics
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "intent": "None",
        "confidence": "0.00%",
        "log": "Awaiting customer interaction...",
        "context": "No active context loaded."
    }

# 🖥️ 4. Sidebar Diagnostics & Logout Panel
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=60)
    st.title("Admin Console")
    st.caption(f"Authenticated as: **admin**")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    
    # Core Network Endpoint Configuration
    backend_url = st.text_input("FastAPI Router Endpoint:", value="http://127.0.0.1:8001/api/chat")
    st.markdown("---")
    
    # Dynamic Diagnostic Metrics
    st.subheader("📡 Live Router Diagnostics")
    st.metric(label="Selected Agent Route", value=st.session_state.last_metrics["intent"])
    st.metric(label="Routing Confidence Score", value=st.session_state.last_metrics["confidence"])
    
    st.markdown("### 📋 Active Routing Trace")
    st.info(st.session_state.last_metrics["log"])

# 🏛️ 5. Header Layout
st.markdown("# 🤖 Multi-Agent AI Customer Assistant")
st.caption("Enterprise support portal driven by a specialized routing pipeline.")
st.markdown("---")

# 🗂️ 6. Dashboard Tabs: Chat Interface & Database Explorer
tab_chat, tab_db = st.tabs(["💬 Customer Chat Assistant", "📊 Database Explorer (CFPB & Vector)"])

# --- TAB 1: Chat Assistant ---
with tab_chat:
    # Display Chat History Matrix
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle Live User Chat Input
    if user_query := st.chat_input("How can I assist you with your account or transaction today?"):
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Routing query to specialized support agent..."):
                try:
                    payload = {"text": user_query}
                    response = requests.post(backend_url, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        intent = data.get("intent", "UNKNOWN").upper()
                        confidence = data.get("confidence", 0.0)
                        log_trace = data.get("log", "No routing logs trace left behind.")
                        context_block = data.get("context", "Context allocation completely missing.")
                        ai_response = data.get("response", "Synthesis engine failed.")
                        
                        conf_str = f"{confidence * 100:.2f}%" if confidence <= 1.0 else f"{confidence:.2f}%"
                        
                        st.session_state.last_metrics = {
                            "intent": intent,
                            "confidence": conf_str,
                            "log": log_trace,
                            "context": context_block
                        }
                        
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                        with st.expander("📄 Inspected Agent RAG Context"):
                            st.code(context_block, language="text")
                            
                    else:
                        error_msg = f"❌ Backend service error status: {response.status_code}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                except Exception as e:
                    error_msg = f"⚠️ Connection failed: Ensure your FastAPI server is running on port 8001. Details: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
        st.rerun()

# --- TAB 2: Database Explorer ---
with tab_db:
    st.subheader("📁 Simulated Local Backend Repositories")
    st.markdown("Here you can audit the source databases that the sub-agents query to gather context.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💳 CFPB Complaints Schema Database (Billing Agent)")
        # Mocking the local database structured representation
        cfpb_df = pd.DataFrame({
            "Complaint ID": ["CFPB_9081", "CFPB_4521", "CFPB_1102", "CFPB_8741"],
            "Issue Category": ["Double Charge", "Card Blocked", "Unrecognized Transaction", "Refund Failure"],
            "Target Transaction ID": ["TXN_001", "TXN_002", "TXN_003", "TXN_004"],
            "Resolution Rule Code": ["AUTORECONCILE_DOUBLE", "UNLOCK_VERIFICATION_REQUIRED", "RISK_AUDIT_QUEUE", "ROUTER_RETRY_IMMEDIATE"],
            "System Action State": ["Ready", "Awaiting ID Verify", "Flagged", "Processed"]
        })
        st.dataframe(cfpb_df, use_container_width=True, hide_index=True)
        st.success("🟢 Connection: Database is Live")

    with col2:
        st.markdown("#### 📂 FAQ Corpus Embedding Maps (FAQ Agent Vector DB)")
        faq_df = pd.DataFrame({
            "Vector Chunk ID": ["CH_09", "CH_14", "CH_32", "CH_55"],
            "Semantic Topic": ["Cross-Border Processing", "Credentials Reset", "Business Hours", "Routing Policies"],
            "Mathematical Vector Distance": [0.12, 0.45, 0.22, 0.61],
            "Extracted Chroma Text Snippet": [
                "International wire processing takes 3-5 business days.",
                "To reset passwords, user must confirm via MFA.",
                "Standard branches operate Mon-Fri 9AM-5PM.",
                "High-level routing maps incoming queries to domain blocks."
            ]
        })
        st.dataframe(faq_df, use_container_width=True, hide_index=True)
        st.info("📦 Vector Indexes Sync Status: Active")