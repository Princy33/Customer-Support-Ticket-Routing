import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from backend import predict_ticket


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f5f8ff 0%,
        #eef3ff 45%,
        #f8f4ff 100%
    );
}

/* Remove default top space */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #07142f 0%,
        #101d45 55%,
        #17265a 100%
    );
    border-right: 1px solid #293b70;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 3px;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #aebce0 !important;
    line-height: 1.5;
}

.sidebar-line {
    height: 1px;
    background: rgba(255,255,255,0.18);
    margin: 25px 0;
}

/* Highlight selected navigation item */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 12px;
    padding: 8px 12px;
    margin: 4px 0;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10);
    border-color: rgba(255,255,255,0.14);
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #176cf5 0%, #6638e8 100%);
    border-color: rgba(255,255,255,0.22);
    box-shadow: 0 5px 14px rgba(0,0,0,0.18);
}

/* Make Plotly chart text readable on the light dashboard */
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .axis-title,
.js-plotly-plot .plotly .legendtext,
.js-plotly-plot .plotly .textpoint {
    fill: #172033 !important;
}

/* Hero */
.hero {
    background: linear-gradient(
        110deg,
        #12347c 0%,
        #155bd1 48%,
        #633cf0 100%
    );
    border-radius: 22px;
    padding: 32px 38px;
    color: white;
    box-shadow: 0 15px 35px rgba(28, 67, 160, 0.25);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -100px;
    top: -150px;
    background: rgba(255,255,255,0.10);
    border-radius: 50%;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 35px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    font-size: 18px;
    font-weight: 600;
    margin-top: 7px;
}

.hero-description {
    font-size: 14px;
    line-height: 1.7;
    max-width: 720px;
    margin-top: 12px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.95);
    border: 1px solid #dbe4f5;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 7px 22px rgba(31, 54, 99, 0.08);
    margin-bottom: 18px;
}

.card-title {
    color: #172b65;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 8px;
}

.card-text {
    color: #64708a;
    font-size: 13px;
    line-height: 1.6;
}

/* Section heading */
.section-title {
    color: #15275c;
    font-size: 23px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Metric cards */
.metric-box {
    background: white;
    border-radius: 17px;
    padding: 20px;
    border: 1px solid #dce5f5;
    box-shadow: 0 6px 18px rgba(35, 57, 105, 0.08);
    min-height: 120px;
}

.metric-icon {
    font-size: 25px;
}

.metric-label {
    font-size: 12px;
    color: #6d7890;
    margin-top: 5px;
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #17377d;
    margin-top: 3px;
}

.metric-small {
    font-size: 11px;
    color: #7b879e;
}

/* Result */
.result-box {
    background: linear-gradient(
        135deg,
        #edf5ff,
        #f5efff
    );
    border: 1px solid #cbd8f5;
    border-radius: 18px;
    padding: 22px;
    margin-top: 18px;
}

.result-title {
    font-size: 20px;
    font-weight: 800;
    color: #162c66;
}

.result-value {
    font-size: 25px;
    font-weight: 800;
    color: #2458d8;
}

/* Priority */
.priority-high {
    background: #ffe8ed;
    color: #c51f45;
    padding: 7px 15px;
    border-radius: 20px;
    font-weight: 800;
}

.priority-medium {
    background: #fff3d8;
    color: #a86a00;
    padding: 7px 15px;
    border-radius: 20px;
    font-weight: 800;
}

.priority-low {
    background: #e4f8ec;
    color: #168448;
    padding: 7px 15px;
    border-radius: 20px;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #1677f5,
        #6a35ef
    );
    color: white;
    font-weight: 700;
    padding: 11px 20px;
    box-shadow: 0 5px 15px rgba(58, 75, 220, 0.22);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(58, 75, 220, 0.32);
}

/* Text area */
textarea {
    border-radius: 12px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #75819a;
    font-size: 11px;
    padding: 25px 0 5px 0;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY CHART STYLE
# ============================================================

def style_chart(fig, title=None, x_title=None, y_title=None):
    """Keep the original dashboard charts, but force readable labels/text."""
    fig.update_layout(
        template="plotly_white",
        title=title if title else fig.layout.title.text,
        title_font=dict(size=20, color="#15275c", family="Arial"),
        font=dict(size=13, color="#172033", family="Arial"),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        margin=dict(l=20, r=35, t=70, b=65),
        xaxis=dict(
            title=x_title if x_title is not None else fig.layout.xaxis.title.text,
            title_font=dict(size=13, color="#172033"),
            tickfont=dict(size=12, color="#172033"),
            showline=True,
            linecolor="#6b7280",
            linewidth=1.2,
            gridcolor="#d9e0ec",
            zeroline=False,
            automargin=True
        ),
        yaxis=dict(
            title=y_title if y_title is not None else fig.layout.yaxis.title.text,
            title_font=dict(size=13, color="#172033"),
            tickfont=dict(size=12, color="#172033"),
            showline=True,
            linecolor="#6b7280",
            linewidth=1.2,
            gridcolor="#d9e0ec",
            zeroline=False,
            automargin=True
        ),
        legend=dict(
            font=dict(size=12, color="#172033"),
            bgcolor="rgba(255,255,255,0.80)"
        ),
        hoverlabel=dict(
            bgcolor="#172033",
            font=dict(color="white", size=12)
        ),
    )
    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div style="font-size:42px;">🤖</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-title">Customer Support AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Intelligent Ticket Routing & Priority Ranking'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🎫 Analyze Ticket",
            "📊 Dataset Insights",
            "📈 Model Performance",
            "🕘 Prediction History",
            "📘 Project Information"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🤖 AI Models")

    st.markdown("🌲 **Random Forest**")
    st.caption("TF-IDF based ticket routing")

    st.markdown("🧠 **GRU + Attention**")
    st.caption("Deep learning routing & priority")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-badge">
🤖 AI-POWERED CUSTOMER SUPPORT
</div>

<div class="hero-title">
🎫 Customer Support Ticket Routing
</div>

<div class="hero-subtitle">
GRU Networks with Attention-Based Priority Ranking
</div>

<div class="hero-description">
Automatically route customer support tickets to the appropriate
support queue and identify ticket priority using Natural Language
Processing, Machine Learning and Deep Learning.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Project Dashboard</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-icon">🎫</div>
        <div class="metric-label">Total English Tickets</div>
        <div class="metric-value">16,338</div>
        <div class="metric-small">Dataset used for modelling</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-icon">📂</div>
        <div class="metric-label">Support Queues</div>
        <div class="metric-value">10</div>
        <div class="metric-small">Routing categories</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-icon">⚡</div>
        <div class="metric-label">Priority Classes</div>
        <div class="metric-value">3</div>
        <div class="metric-small">Low • Medium • High</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-icon">🤖</div>
        <div class="metric-label">AI Models</div>
        <div class="metric-value">5</div>
        <div class="metric-small">4 ML + 1 GRU architecture</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">⚙️ How It Works</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="card">
        <div style="font-size:30px;">🎫</div>
        <div class="card-title">1. Enter Ticket</div>
        <div class="card-text">
        Customer enters the subject and problem description.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
        <div style="font-size:30px;">🧠</div>
        <div class="card-title">2. AI Analysis</div>
        <div class="card-text">
        NLP preprocessing and trained ML/DL models analyse
        the ticket content.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="card">
        <div style="font-size:30px;">🎯</div>
        <div class="card-title">3. Smart Routing</div>
        <div class="card-text">
        The system predicts the support queue, priority and
        urgency score.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # QUICK ANALYZE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">⚡ Quick Ticket Analysis</div>',
        unsafe_allow_html=True
    )

    with st.container():

        subject = st.text_input(
            "Ticket Subject",
            placeholder="Example: Unable to login to my account"
        )

        description = st.text_area(
            "Customer Problem / Description",
            placeholder="Describe the customer's problem in detail...",
            height=130
        )

        if st.button("🚀 Analyze Ticket", use_container_width=True):

            if not description.strip() and not subject.strip():

                st.warning(
                    "Please enter a ticket subject or description."
                )

            else:

                full_text = f"{subject} {description}".strip()

                with st.spinner("AI is analysing the ticket..."):

                    result = predict_ticket(full_text)

                st.session_state.history.append({
                    "Ticket": subject if subject else description[:45],
                    "Queue": result["gru_queue"],
                    "Priority": result["priority"],
                    "Urgency": round(result["urgency_score"], 2),
                    "Time": datetime.now().strftime("%I:%M %p")
                })

                st.success("Ticket analysed successfully!")

                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    st.metric(
                        "Recommended Queue",
                        result["gru_queue"]
                    )

                with r2:
                    st.metric(
                        "Priority",
                        result["priority"].upper()
                    )

                with r3:
                    st.metric(
                        "Urgency Score",
                        f"{result['urgency_score']:.1f}/100"
                    )

                with r4:
                    st.metric(
                        "RF Confidence",
                        f"{result['random_forest_confidence']*100:.1f}%"
                    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Model Performance</div>',
        unsafe_allow_html=True
    )

    performance = pd.DataFrame({
        "Model": [
            "Naive Bayes",
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "GRU + Attention"
        ],
        "Accuracy": [
            40.85,
            49.20,
            56.95,
            67.93,
            43.60
        ]
    })

    fig = px.bar(
        performance,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Model Accuracy Comparison",
        range_y=[0, 80]
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont=dict(size=13, color="#172033"),
        cliponaxis=False,
        marker_line_width=0.8
    )

    style_chart(
        fig,
        title="Model Accuracy Comparison",
        x_title="Models",
        y_title="Accuracy (%)"
    )
    fig.update_yaxes(range=[0, 80], dtick=20)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ANALYZE TICKET PAGE
# ============================================================

elif page == "🎫 Analyze Ticket":

    st.markdown(
        '<div class="section-title">🎫 AI Ticket Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
    <div class="card-title">🤖 Intelligent Ticket Classification</div>
    <div class="card-text">
    Enter a customer support issue below. The trained models will
    analyse the ticket and recommend the support queue and priority.
    </div>
    </div>
    """, unsafe_allow_html=True)

    subject = st.text_input(
        "Ticket Subject",
        placeholder="Example: Payment failed"
    )

    description = st.text_area(
        "Customer Problem",
        placeholder="Describe the customer's issue...",
        height=180
    )

    if st.button("🔍 Analyze Customer Ticket", use_container_width=True):

        if not subject.strip() and not description.strip():

            st.error("Please enter the ticket details.")

        else:

            ticket = f"{subject} {description}"

            with st.spinner("Running Random Forest and GRU + Attention models..."):

                result = predict_ticket(ticket)

            st.markdown(
                '<div class="section-title">🎯 Prediction Results</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("""
                <div class="result-box">
                <div class="result-title">
                🌲 Random Forest Routing
                </div>
                </div>
                """, unsafe_allow_html=True)

                st.write(
                    "**Predicted Queue:**",
                    result["random_forest_queue"]
                )

                st.progress(
                    result["random_forest_confidence"]
                )

                st.caption(
                    f"Confidence: "
                    f"{result['random_forest_confidence']*100:.2f}%"
                )

            with col2:

                st.markdown("""
                <div class="result-box">
                <div class="result-title">
                🧠 GRU + Attention Routing
                </div>
                </div>
                """, unsafe_allow_html=True)

                st.write(
                    "**Predicted Queue:**",
                    result["gru_queue"]
                )

                st.progress(
                    result["gru_confidence"]
                )

                st.caption(
                    f"Confidence: "
                    f"{result['gru_confidence']*100:.2f}%"
                )

            st.markdown(
                '<div class="section-title">⚡ Priority Ranking</div>',
                unsafe_allow_html=True
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Predicted Priority",
                    result["priority"].upper()
                )

            with p2:
                st.metric(
                    "Priority Confidence",
                    f"{result['priority_confidence']*100:.2f}%"
                )

            with p3:
                st.metric(
                    "Urgency Score",
                    f"{result['urgency_score']:.2f}/100"
                )

            if result["urgency_score"] >= 70:
                st.error(
                    "🔴 High urgency ticket — should be handled quickly."
                )

            elif result["urgency_score"] >= 40:
                st.warning(
                    "🟡 Medium urgency ticket."
                )

            else:
                st.success(
                    "🟢 Low urgency ticket."
                )


# ============================================================
# DATASET INSIGHTS
# ============================================================

elif page == "📊 Dataset Insights":

    st.markdown(
        '<div class="section-title">📊 Dataset Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
    <div class="card-title">📁 Dataset Overview</div>
    <div class="card-text">
    The project uses the English subset of the Customer IT Support
    Ticket Dataset. The dataset contains customer support tickets
    with queue and priority information.
    </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("English Tickets", "16,338")
    c2.metric("Support Queues", "10")
    c3.metric("Priority Classes", "3")

    queue_data = pd.DataFrame({
        "Queue": [
            "Technical Support",
            "Product Support",
            "Customer Service",
            "IT Support",
            "Billing and Payments",
            "Returns and Exchanges",
            "Service Outages and Maintenance",
            "Sales and Pre-Sales",
            "Human Resources",
            "General Inquiry"
        ],
        "Tickets": [
            4737,
            3073,
            2410,
            1942,
            1595,
            820,
            664,
            513,
            348,
            236
        ]
    })

    fig = px.bar(
        queue_data,
        x="Tickets",
        y="Queue",
        orientation="h",
        title="Support Queue Distribution"
    )

    fig.update_traces(
        text=queue_data["Tickets"],
        texttemplate="%{text:,}",
        textposition="outside",
        textfont=dict(size=12, color="#172033"),
        cliponaxis=False,
        marker_line_width=0.8
    )

    style_chart(
        fig,
        title="Support Queue Distribution",
        x_title="Number of Tickets",
        y_title="Support Queue"
    )
    fig.update_xaxes(showgrid=True, dtick=1000)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    priority_data = pd.DataFrame({
        "Priority": ["Medium", "High", "Low"],
        "Tickets": [6618, 6346, 3374]
    })

    fig2 = px.pie(
        priority_data,
        names="Priority",
        values="Tickets",
        hole=0.45,
        title="Priority Distribution"
    )

    fig2.update_traces(
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=13, color="#172033"),
        marker_line=dict(color="#ffffff", width=2)
    )
    style_chart(fig2, title="Priority Distribution")

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.markdown(
        '<div class="section-title">📈 Model Performance</div>',
        unsafe_allow_html=True
    )

    performance = pd.DataFrame({
        "Model": [
            "Naive Bayes",
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "GRU + Attention"
        ],
        "Accuracy": [
            40.85,
            49.20,
            56.95,
            67.93,
            43.60
        ]
    })

    fig = px.bar(
        performance,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Accuracy Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont=dict(size=13, color="#172033"),
        cliponaxis=False,
        marker_line_width=0.8
    )

    style_chart(
        fig,
        title="Accuracy Comparison",
        x_title="Models",
        y_title="Accuracy (%)"
    )
    fig.update_yaxes(range=[0, 80], dtick=20)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
    <div class="card">

    <div class="card-title">
    🏆 Best Traditional Model
    </div>

    <div class="card-text">
    Random Forest achieved the highest routing accuracy among
    the traditional machine learning models with approximately
    67.93% accuracy.
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "🕘 Prediction History":

    st.markdown(
        '<div class="section-title">🕘 Prediction History</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions yet. Go to Analyze Ticket and analyse a customer ticket."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear Prediction History"):

            st.session_state.history = []

            st.success("Prediction history cleared.")

            st.rerun()


# ============================================================
# PROJECT INFORMATION
# ============================================================

elif page == "📘 Project Information":

    st.markdown(
        '<div class="section-title">📘 Project Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <div class="card-title">
    🎯 Project Objective
    </div>

    <div class="card-text">

    The objective of this project is to automatically classify
    customer support tickets into appropriate support queues and
    determine their priority using Natural Language Processing,
    Machine Learning and Deep Learning.

    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🔄 NLP Pipeline</div>',
        unsafe_allow_html=True
    )

    pipeline = pd.DataFrame({
        "Step": [
            "1. Input Ticket",
            "2. Text Preprocessing",
            "3. Feature Extraction",
            "4. Machine Learning",
            "5. GRU + Attention",
            "6. Priority Ranking",
            "7. Final Prediction"
        ],
        "Process": [
            "Customer enters support problem",
            "Lowercase, URL removal, stopword removal and lemmatization",
            "TF-IDF representation for traditional ML",
            "Naive Bayes, Logistic Regression, SVM and Random Forest",
            "Sequence modelling with GRU and Attention",
            "High-priority probability converted to urgency score",
            "Support queue + priority + confidence"
        ]
    })

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">🧠 Technologies Used</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        <div class="card">
        <div class="card-title">🐍 Python</div>
        <div class="card-text">
        Main programming language used for data processing,
        modelling and application development.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
        <div class="card">
        <div class="card-title">🧠 NLP & ML</div>
        <div class="card-text">
        NLTK, TF-IDF, Scikit-learn and Random Forest are used
        for text processing and ticket classification.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tech3:
        st.markdown("""
        <div class="card">
        <div class="card-title">🤖 Deep Learning</div>
        <div class="card-text">
        GRU networks with Attention are used for advanced
        ticket routing and priority prediction.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🚀 Future Scope</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <div class="card-text">

    • Multilingual ticket classification<br>
    • Real-time integration with customer support systems<br>
    • Larger transformer-based NLP models<br>
    • Automatic agent assignment<br>
    • Continuous model retraining<br>
    • Email and chatbot integration

    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
❤️ Customer Support Ticket Routing using GRU Networks with Attention-Based Priority Ranking
&nbsp; | &nbsp;
TYBSc Data Science Mini Project
&nbsp; | &nbsp;
Built with Streamlit
</div>
""", unsafe_allow_html=True)