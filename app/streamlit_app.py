"""Football Tactics Q&A - Streamlit interface."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.rag import FootballTacticsRAG, PROMPT_TEMPLATES
from app.db import save_feedback, get_recent_conversations, get_dashboard_stats, get_chart_data


st.set_page_config(page_title="Football Tactics Q&A", layout="wide")

# Sidebar
st.sidebar.title("Settings")
prompt_template = st.sidebar.selectbox(
    "Prompt style",
    options=list(PROMPT_TEMPLATES.keys()),
    index=1,  # default to "expert"
)
page = st.sidebar.radio("Page", ["Ask", "Dashboard"])


@st.cache_resource
def load_rag():
    return FootballTacticsRAG()


# --- Ask page ---
if page == "Ask":
    st.title("Football Tactics Q&A")
    st.caption(
        "Ask about formations, pressing systems, tactical concepts, "
        "match analysis, and playing styles."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is gegenpressing and which teams use it?",
    )

    if st.button("Ask", type="primary") and question:
        rag = load_rag()

        with st.spinner("Thinking..."):
            result = rag.answer(question, prompt_template=prompt_template)

        # Answer
        st.markdown("### Answer")
        st.markdown(result["answer"])

        # Relevance badge
        if result["relevance"]:
            color_map = {
                "RELEVANT": "green",
                "PARTLY_RELEVANT": "orange",
                "NON_RELEVANT": "red",
            }
            color = color_map.get(result["relevance"], "gray")
            st.markdown(
                f'Auto-judge: :{color}[{result["relevance"]}]'
            )

        # Feedback buttons
        col1, col2, _ = st.columns([1, 1, 8])
        conv_id = result.get("conversation_id")

        with col1:
            if st.button("👍", key="up"):
                if conv_id:
                    save_feedback(conv_id, 1)
                    st.success("Thanks!")
        with col2:
            if st.button("👎", key="down"):
                if conv_id:
                    save_feedback(conv_id, -1)
                    st.warning("Noted, thanks.")

        # Details
        with st.expander("Retrieved Sources"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(f"**{i}. {src['title']}** ({src['source']})")
                st.text(src["content"][:300] + "...")
                st.divider()

        with st.expander("Query Details"):
            st.json(
                {
                    "rewritten_query": result["rewritten_query"],
                    "model": result["model"],
                    "prompt_template": result["prompt_template"],
                    "response_time_s": round(result["response_time"], 2),
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "cost_usd": round(result["cost"], 6),
                }
            )

# --- Dashboard page ---
elif page == "Dashboard":
    st.title("Monitoring Dashboard")
    from app.db import _get_backend
    db_label = "PostgreSQL" if _get_backend() == "postgres" else "SQLite"
    st.caption(f"Live stats from {db_label}.")

    try:
        stats = get_dashboard_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", stats["total_queries"])
        col2.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
        col3.metric("Relevance %", f"{stats['relevance_pct']}%")
        col4.metric("Total Cost", f"${stats['total_cost']:.4f}")

        # Charts
        st.divider()
        try:
            import pandas as pd
            charts = get_chart_data()

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Response Time")
                if charts["response_times"]:
                    df_rt = pd.DataFrame(charts["response_times"])
                    df_rt["created_at"] = pd.to_datetime(df_rt["created_at"])
                    st.line_chart(df_rt, x="created_at", y="response_time")
                else:
                    st.info("No data yet.")

            with col_right:
                st.subheader("Relevance Distribution")
                if charts["relevance_dist"]:
                    df_rel = pd.DataFrame(
                        list(charts["relevance_dist"].items()),
                        columns=["Relevance", "Count"],
                    )
                    st.bar_chart(df_rel, x="Relevance", y="Count")
                else:
                    st.info("No data yet.")

            col_left2, col_right2 = st.columns(2)

            with col_left2:
                st.subheader("User Feedback")
                if sum(charts["feedback_dist"].values()) > 0:
                    df_fb = pd.DataFrame(
                        list(charts["feedback_dist"].items()),
                        columns=["Type", "Count"],
                    )
                    st.bar_chart(df_fb, x="Type", y="Count")
                else:
                    st.info("No feedback yet.")

            with col_right2:
                st.subheader("Token Usage")
                if charts["token_usage"]:
                    df_tok = pd.DataFrame(charts["token_usage"])
                    df_tok["created_at"] = pd.to_datetime(df_tok["created_at"])
                    st.line_chart(df_tok, x="created_at", y=["prompt_tokens", "completion_tokens"])
                else:
                    st.info("No data yet.")

            if charts["model_usage"]:
                st.subheader("Model Usage")
                df_model = pd.DataFrame(
                    list(charts["model_usage"].items()),
                    columns=["Model", "Queries"],
                )
                st.bar_chart(df_model, x="Model", y="Queries")
        except Exception:
            pass

        st.divider()
        st.subheader("Recent Conversations")
        convos = get_recent_conversations(20)
        if convos:
            st.dataframe(
                convos,
                column_config={
                    "id": st.column_config.NumberColumn("ID"),
                    "question": st.column_config.TextColumn("Question", width="large"),
                    "relevance": st.column_config.TextColumn("Relevance"),
                    "response_time": st.column_config.NumberColumn("Time (s)", format="%.2f"),
                    "cost": st.column_config.NumberColumn("Cost ($)", format="%.6f"),
                },
                hide_index=True,
            )
        else:
            st.info("No conversations yet. Ask some questions first!")
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        st.info("Make sure the database is available.")
