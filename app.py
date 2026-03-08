import streamlit as st
import os
import json
import hashlib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="InsightFlow",
    page_icon="📊",
    layout="wide"
)

# =====================================
# PATH SETUP
# =====================================

base_dir = os.path.dirname(os.path.abspath(__file__))
users_file = os.path.join(base_dir, "users.json")

# logo path
logo_path = os.path.join(base_dir, "assets", "logo.png")

# =====================================
# SESSION STATE INIT
# =====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "ds_test_passed" not in st.session_state:
    st.session_state.ds_test_passed = False

# =====================================
# USER DATABASE FUNCTIONS
# =====================================

def load_users():
    if os.path.exists(users_file):
        with open(users_file, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(users_file, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================
# LOAD CSS
# =====================================

def load_css():
    css_path = os.path.join(base_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =====================================
# REGISTER PAGE
# =====================================

def register_page():

    st.subheader("📝 Create Account")

    name = st.text_input("Full Name")
    email = st.text_input("Email ID", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_pass")
    confirm_password = st.text_input("Confirm Password")

    if st.button("Create Account"):

        users = load_users()

        if not email.endswith("@gmail.com"):
            st.error("❌ Email must end with @gmail.com")
            return

        if email in users:
            st.error("⚠ Account already exists")
            return

        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return

        users[email] = {
            "name": name,
            "password": hash_password(password)
        }

        save_users(users)

        st.success("✅ Account created successfully! Please login.")

# =====================================
# LOGIN PAGE
# =====================================

def login_page():

    st.title("🔐 Login to InsightFlow")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:

        email = st.text_input("Email ID", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):

            users = load_users()

            if email not in users:
                st.error("❌ Account does not exist")
                return

            if users[email]["password"] != hash_password(password):
                st.error("❌ Incorrect password")
                return

            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_name = users[email]["name"]

            st.success("✅ Login successful")
            st.rerun()

    with tab2:
        register_page()

# =====================================
# DATA SCIENTIST TEST
# =====================================

def data_scientist_test():

    st.subheader("🧠 Data Scientist Eligibility Test")
    st.info("Minimum 3/5 required to unlock Data Scientist role.")

    score = 0

    q1 = st.radio("1️⃣ What is Overfitting?", [
        "Model performs well on training but poorly on test data",
        "Model performs well on all data",
        "Model has too few parameters",
        "Data has missing values"
    ])

    q2 = st.radio("2️⃣ Which library is used for Explainable AI?", [
        "NumPy", "Pandas", "SHAP", "Seaborn"
    ])

    q3 = st.radio("3️⃣ Purpose of Cross Validation?", [
        "Increase dataset size",
        "Evaluate model stability",
        "Remove outliers",
        "Normalize data"
    ])

    q4 = st.radio("4️⃣ Best metric for imbalanced classification?", [
        "Accuracy", "F1-Score", "Mean Squared Error", "R2 Score"
    ])

    q5 = st.radio("5️⃣ SHAP stands for:", [
        "SHapley Additive exPlanations",
        "Statistical Hierarchical Analysis Process",
        "Shallow Hyperparameter Algorithm",
        "None of the above"
    ])

    if st.button("Submit Eligibility Test"):

        if q1 == "Model performs well on training but poorly on test data":
            score += 1
        if q2 == "SHAP":
            score += 1
        if q3 == "Evaluate model stability":
            score += 1
        if q4 == "F1-Score":
            score += 1
        if q5 == "SHapley Additive exPlanations":
            score += 1

        if score >= 3:
            st.success(f"✅ Passed! Score: {score}/5")
            st.session_state.ds_test_passed = True
        else:
            st.error(f"❌ Failed! Score: {score}/5")
            st.session_state.ds_test_passed = False

# =====================================
# ROLE SELECTION
# =====================================

def role_selection_page():

    st.title("👤 Select Your Role")

    role = st.selectbox("Choose your role", [
        "Non-Technical User",
        "Business User",
        "Data Analyst",
        "Data Scientist"
    ])

    if role == "Data Scientist":
        data_scientist_test()

    if st.button("Continue"):

        if role == "Data Scientist" and not st.session_state.ds_test_passed:
            st.warning("⚠ You must pass the eligibility test.")
            return

        st.session_state.user_role = role
        st.success("✅ Role Selected Successfully!")
        st.rerun()

# =====================================
# HIDE SIDEBAR BEFORE AUTH
# =====================================

if not st.session_state.logged_in or st.session_state.user_role is None:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

# =====================================
# AUTH FLOW
# =====================================

if not st.session_state.logged_in:
    login_page()
    st.stop()

if st.session_state.user_role is None:
    role_selection_page()
    st.stop()

# =====================================
# CUSTOM HEADER
# =====================================

header_col1, header_col2 = st.columns([6,2])

# LEFT SIDE
with header_col1:

    left_inner1, left_inner2 = st.columns([1,5])

    with left_inner1:

        st.markdown("<div style='margin-top:15px'></div>", unsafe_allow_html=True)

        if os.path.exists(logo_path):
            st.image(logo_path, width=120)

    with left_inner2:

        st.markdown("""
        <div style="line-height:1.2;">
            <h2 style="margin-bottom:0;">InsightFlow</h2>
            <span style="color:gray; font-size:20px;">
                Unifying Data into Intelligent Decisions
            </span>
        </div>
        """, unsafe_allow_html=True)

# RIGHT SIDE
with header_col2:

    st.markdown(f"""
    <div style="text-align:right;">

    <div style="font-size:40px;font-weight:800;color:#000000;">
    👤 {st.session_state.user_name}
    </div>

    <div style="font-size:14px;color:#94a3b8;">
    📧 {st.session_state.user_email}
    </div>

    <div style="font-size:15px;color:#38bdf8;margin-top:4px;">
    🎯 Role: {st.session_state.user_role}
    </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.user_name = ""
        st.session_state.user_role = None
        st.session_state.ds_test_passed = False

        st.rerun()

st.divider()

st.subheader("Use the sidebar to begin the workflow")

# =====================================
# SIDEBAR NAVIGATION
# =====================================

st.sidebar.title("🚀 InsightFlow Workflow")

st.sidebar.page_link("pages/1_Upload_Dataset.py", label="1️⃣ Upload Dataset")
st.sidebar.page_link("pages/2_Data_Intelligence.py", label="2️⃣ Data Intelligence")
st.sidebar.page_link("pages/3_Data_Cleaning.py", label="3️⃣ Data Cleaning")
st.sidebar.page_link("pages/4_EDA.py", label="4️⃣ EDA")
st.sidebar.page_link("pages/5_Model_Training.py", label="5️⃣ Model Training")
st.sidebar.page_link("pages/6_Explainable_AI.py", label="6️⃣ Explainable AI")
st.sidebar.page_link("pages/7_Prediction.py", label="7️⃣ Prediction")
st.sidebar.page_link("pages/8_Insights.py", label="8️⃣ Business Insights")
st.sidebar.page_link("pages/9_Report_Generation.py", label="9️⃣ Generate Report")

st.markdown("---")

st.caption("© 2026 InsightFlow | AI-Powered Decision Intelligence Platform")

