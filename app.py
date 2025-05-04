import os, pandas as pd, datetime, time
import streamlit as st
from openai import OpenAI
from src.retrieval import AIDocumentStore, embed_query
from src.generator import build_prompt, generate_answer
from src.tts import toggle_speech
from src.memory import add_to_memory, format_memory_prompt
from src.upload_utils import extract_text_from_pdf, extract_text_from_txt, chunk_text, generate_chunk_title

# Initialize history file if missing
if not os.path.exists("data/history.csv"):
    pd.DataFrame(columns=["timestamp", "question", "answer"]).to_csv("data/history.csv", index=False)

# Sidebar settings and API key input
with st.sidebar:
    st.header("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 1000, 300, 50)
    prompt_style = st.selectbox("Prompt Style", [
        "Default", "Concise", "Beginner-Friendly", "Explain Step-by-Step", "With Citations Only"
    ])
    cot_enabled = st.toggle("Chain-of-Thought (Internal Reasoning)",
                            value=False,
                            help="Helps the model think through the problem before answering.")

    st.header("Enter API Key")

    # Reset input box after key update
    if st.session_state.get("clear_api_key_box"):
        st.session_state.api_key_input = ""
        del st.session_state["clear_api_key_box"]

    # Secure text input for OpenAI API key
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="We do not store your API key. It stays in your session.",
        key="api_key_input"
    )

    # Trigger key validation
    if api_key_input and api_key_input != st.session_state.get("openai_api_key", ""):
        st.session_state.openai_api_key = api_key_input
        st.session_state.api_key_valid = False
        st.session_state.validation_complete = False

        status_placeholder = st.empty()

        # Validate by calling the embeddings endpoint
        with st.spinner("Validating API Key..."):
            try:
                OpenAI(api_key=api_key_input).embeddings.create(
                    model="text-embedding-ada-002",
                    input="TEST"
                )
                validation_passed = True
            except Exception:
                validation_passed = False

        # Store result in session state
        if validation_passed:
            st.session_state.api_key_valid = True
            st.session_state.validation_complete = True
            st.session_state.clear_api_key_box = True
            status_placeholder.success("API Key is valid!")
            time.sleep(2.5)
            status_placeholder.empty()
            st.rerun()
        else:
            st.session_state.api_key_valid = False
            st.session_state.validation_complete = False
            status_placeholder.error("Invalid API Key. Please try again.")
            time.sleep(2.5)
            status_placeholder.empty()

# Load FAISS index
@st.cache_resource(show_spinner=False)
def load_ai_knower():
    store = AIDocumentStore("data/arxiv_dataset.csv", "data/faiss.index")
    index = store.load_index()
    store.load_and_split()
    return store, index

store, index = load_ai_knower()
documents = store.get_documents()
metadata = store.get_metadata()

# Session state setup
st.session_state.setdefault("answer", "")
st.session_state.setdefault("matched_docs", [])
st.session_state.setdefault("tts_active", False)
st.session_state.setdefault("qa_memory", [])
st.session_state.setdefault("explanation", "")
st.session_state.setdefault("prompt", "")
st.session_state.setdefault("api_key_valid", False)
st.session_state.setdefault("validation_complete", False)

# Stop app until key is validated
if not st.session_state.api_key_valid and not st.session_state.validation_complete:
    st.stop()

# Page header
st.title("AI Study Buddy")
st.text("Ask an AI/ML related question via text or upload. Get answers with memory, reasoning, and sources!")

# PDF/TXT upload
uploaded_file = st.file_uploader("Optional: Upload a PDF or TXT file", type=["pdf", "txt"])
uploaded_chunks = []
selected_chunk = ""

if uploaded_file:
    try:
        # Extract and chunk document text
        full_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_txt(uploaded_file)
        uploaded_chunks = chunk_text(full_text, chunk_size=300, overlap=20)

        # Generate short titles for each chunk
        @st.cache_data(show_spinner=False)
        def get_chunk_titles(chunks):
            return [generate_chunk_title(chunk) for chunk in chunks]

        chunk_titles = get_chunk_titles(uploaded_chunks)

        # Let user select a section to focus the query
        selected_idx = st.selectbox(
            "Choose a section to include with your question:",
            range(len(chunk_titles)),
            format_func=lambda i: f"Section {i+1}: {chunk_titles[i]}"
        )
        selected_chunk = uploaded_chunks[selected_idx]
        st.success("Document loaded and chunked successfully.")
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

# Question input
with st.form("question_form"):
    query = st.text_area(
        "Enter your question:",
        height=68,
        placeholder="What is the difference between supervised and unsupervised learning?",
        key="user_query"
    )
    submit = st.form_submit_button("Submit")

# Answer generation pipeline
if submit and query:
    # Combine question with selected upload (if applicable)
    input_context = selected_chunk if selected_chunk else ""
    full_input = query if not input_context else f"{input_context}\n{query}"

    with st.spinner("Thinking..."):
        memory_context = format_memory_prompt(st.session_state.qa_memory)
        q_emb = embed_query(full_input).reshape(1, -1)
        _, I = index.search(q_emb, k=3)
        matched_docs = [
            #Truncate to 1000 characters
            (doc[:1000], meta)
            for doc, meta in [(documents[i], metadata[i]) for i in I[0]]
        ]

        # Construct full prompt with style, context, memory, and reasoning
        st.session_state.prompt = build_prompt(
            question=full_input,
            docs_metadata=matched_docs,
            style=prompt_style,
            memory_block=memory_context,
            cot=cot_enabled
        )

        # Ask the model for an answer
        answer = generate_answer(st.session_state.prompt, temperature=temperature, max_tokens=max_tokens)

        # Ask the model to explain the answer
        explanation_prompt = (
            "You are a helpful AI tutor. Briefly explain why the following answer is accurate, "
            "based on the context it was built from.\n\n"
            f"Answer:\n{answer}\n\n"
            f"Context:\n{st.session_state.prompt}\n\n"
            "Explain why this answer makes sense:"
        )
        explanation = generate_answer(explanation_prompt, temperature=0.3, max_tokens=200)

    # Store results in session state and history
    st.session_state.answer = answer
    st.session_state.matched_docs = matched_docs
    st.session_state.qa_memory = add_to_memory(st.session_state.qa_memory, query, answer)
    st.session_state.explanation = explanation
    st.session_state.tts_active = False

    # Append to local history file
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{"timestamp": ts, "question": query, "answer": answer}])
    df.to_csv("data/history.csv", mode="a", header=False, index=False)

# Display answer + extra features
if st.session_state.answer:
    st.subheader("📚 Answer")
    st.write(st.session_state.answer)

    if st.button("🔊 Read Aloud"):
        st.session_state.tts_active = toggle_speech(st.session_state.answer)

    with st.expander("🔗 Sources"):
        for i, (chunk, meta) in enumerate(st.session_state.matched_docs, start=1):
            st.markdown(f"**{i}. [{meta['title']}]({meta['url']})**")

    with st.expander("🧠 Previous Q&A Context"):
        for i, (q, a) in enumerate(st.session_state.qa_memory):
            st.markdown(f"**Q:** {q}  \n**A:** {a}")

    with st.expander("🤔 Explanation (Why This Answer?)"):
        st.write(st.session_state.explanation)

    with st.expander("🧩 Reasoning Trace (Full Prompt)"):
        st.code(st.session_state.prompt)

    with st.expander("📜 History"):
        num_history_to_show = st.number_input("How many recent Q&As to display?", min_value=1, max_value=20, value=5, step=1)
        if os.path.exists("data/history.csv"):
            hist = pd.read_csv("data/history.csv")
            hist_to_show = hist.tail(num_history_to_show).iloc[::-1]

            # Two column layout for downloads
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download as CSV",
                    data=hist_to_show.to_csv(index=False).encode("utf-8"),
                    file_name="ai_study_buddy_history.csv",
                    mime="text/csv"
                )
            with col2:
                txt_log = "\n\n".join(
                    f"[{row['timestamp']}]\nQ: {row['question']}\nA: {row['answer']}"
                    for _, row in hist_to_show.iterrows()
                )
                st.download_button(
                    label="Download as TXT",
                    data=txt_log,
                    file_name="ai_study_buddy_history.txt",
                    mime="text/plain"
                )

            for idx, row in hist_to_show.iterrows():
                st.markdown(f"**{row['timestamp']}**  \n**Q:** {row['question']}  \n**A:** {row['answer']}\n")
                if st.button(f"🔊 Read Answer {idx+1}", key=f"tts_history_{idx}"):
                    st.session_state.tts_active = toggle_speech(row['answer'])

st.markdown("---")
st.markdown("""
📧 Questions or feedback? Reach out at [mia5663@psu.edu](mailto:mia5663@psu.edu)  
💼 Connect with me on [LinkedIn](https://www.linkedin.com/in/mohammedalbattah)
""")