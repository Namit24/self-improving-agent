import streamlit as st
import pandas as pd
import json
from dsa_agent import DSAAgent

st.set_page_config(page_title="Self-improving Agent", layout="wide")

st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Source Code Pro', monospace; }
    .stCodeBlock { border-radius: 10px; border: 1px solid #444; }
    .stMetric { background-color: #0e1117; border: 1px solid #333; }
    code { font-family: 'Source Code Pro', monospace !important; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("Self-Improving Code Agent")
st.caption("Enter any problem. The agent will write tests, write code, and optimize it live.")

with st.container():
    col_input, col_config = st.columns([3, 1])
    with col_input:
        user_problem = st.text_area(
            "Problem Statement", 
            height=150, 
            placeholder="E.g., Write a function that takes a list of integers and returns the length of the longest consecutive sequence."
        )
    with col_config:
        st.write("### Settings")
        st.info("Using **Gemini 2.5 Flash**\n\nStreaming: **ON**\n\nAuto-QA: **ON**")

if st.button("Initialize Agent & Solve", type="primary", disabled=not user_problem):
    
    agent = DSAAgent(user_problem)
    
    col_main, col_stats = st.columns([1.3, 0.7])
    
    with col_main:
        st.subheader("Live Workspace")
        code_placeholder = st.empty()
        
    with col_stats:
        st.subheader("Agent Internals")
        status_box = st.empty()
        tests_expander = st.expander("Generated Test Cases", expanded=False)
        metrics_container = st.container()
        chart_placeholder = st.empty()
        logs_expander = st.expander("Execution Logs", expanded=True)
        log_text_placeholder = logs_expander.empty()
        
    history_data = []
    
    for step in agent.solve_generator():
        
        if step.status == 'streaming':
            code_placeholder.code(step.code + " ▌", language='python')
            continue
            
        if step.status in ['starting', 'improving', 'warning']:
            status_box.info(f"{step.logs}")
            
        if step.status == 'tests_generated':
            status_box.success("Test Cases Generated")
            tests_expander.json(agent.test_cases)
            
        if step.status == 'complete':
            code_placeholder.code(step.code, language='python')
            
            history_data.append({
                "Iteration": step.iteration,
                "Reward": step.reward,
                "Time (ms)": step.avg_time_ms
            })
            
            if len(history_data) > 0:
                df = pd.DataFrame(history_data)
                chart_placeholder.line_chart(df.set_index("Iteration")[['Reward']])
            
            with metrics_container:
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{int(step.correctness * 100)}%")
                c2.metric("Latency", f"{step.avg_time_ms:.3f}ms")
                c3.metric("Big-O", step.complexity)
            
            logs_text = "\n".join(step.test_breakdown) if step.test_breakdown else step.logs
            log_text_placeholder.code(logs_text, language="text")

        if step.status == 'finished':
            code_placeholder.code(step.code, language='python')
            status_box.success(f"d{step.logs}")
            st.balloons()
            break
            
    if step.status != 'finished' and step.status != 'error':
        st.warning("Maximum iterations reached.")