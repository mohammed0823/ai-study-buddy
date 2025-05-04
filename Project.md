# AI Study Buddy – Final Project Report

**CMPSC 441 – Artificial Intelligence**  
**Author:** Mohammed Albattah  
**Professor:** Dr. Pulin Agrawal  

## Section 1: Base System Functionality

**AI Study Buddy** is an interactive AI-powered educational tool that uses retrieval-augmented generation (RAG) to help users explore topics in artificial intelligence and machine learning. It supports the following core scenarios:

### Supported Scenarios:
- Answering natural language questions about AI/ML using RAG.
- Uploading a custom PDF or TXT file and querying over it.
- Selecting a section of uploaded content for focused Q&A.
- Remembering recent Q&A pairs to enhance conversational context.
- Adjusting model behavior using temperature and max token sliders.
- Exporting conversation history as CSV or TXT.
- Hearing AI answers read aloud using text-to-speech (TTS).
- Viewing the reasoning trace and AI explanation for each answer.

The system is fully functional, tested, and runs without errors in Streamlit. This directly supports LO1 and LO3, applying AI methods in a working, modular system.

---

## Section 2: Prompt Engineering and Model Parameter Choice

The system uses prompt engineering to tailor answers to user preferences and comprehension levels:

- **Prompt Style Options**:
  - Default: Direct and informative.
  - Concise: Short, focused answers.
  - Beginner-Friendly: Simple, accessible language.
  - Explain Step-by-Step: Clear, ordered reasoning.
  - With Citations Only: Forces grounding in provided documents.

- **Parameter Settings**:
  - Temperature slider (0.0–1.0) to control creativity.
  - Max tokens slider (100–1000) to cap output length.

These features give users meaningful control over the AI's tone and structure, while ensuring responses remain grounded in reliable context (LO1).

---

## Section 3: Tools Usage

The project integrates several tools and libraries from the Python ecosystem:

- [FAISS](https://github.com/facebookresearch/faiss) – For document vector indexing and similarity search.
- [PyMuPDF](https://pymupdf.readthedocs.io/) – To parse text from uploaded PDFs.
- [pyttsx3](https://pyttsx3.readthedocs.io/) – For offline-capable TTS playback.
- [Streamlit](https://streamlit.io/) – For UI, state management, and API key handling.
- [pytest + coverage](https://docs.pytest.org/) – For automated testing and coverage reporting.
- [OpenAI API](https://openai.com/) – For both embeddings and completions.

These tools work together within a modular architecture, supporting LO1 and LO2.

---

## Section 4: Planning & Reasoning

The system incorporates multi-step reasoning through:

- A toggle option for "Chain-of-Thought" prompting, which encourages LLMs to reason before arriving at a final answer.
- An **Explanation Panel** that uses a second LLM call to summarize why the main response is correct.
- A **Reasoning Trace** view that shows the full constructed prompt.

These features increase transparency in how the AI reaches its conclusions, supporting LO1's emphasis on AI planning techniques.

---

## Section 5: RAG Implementation

The system uses a hybrid RAG approach to provide accurate, grounded answers:

- It embeds a local Arxiv dataset (`arxiv_dataset.csv`) using OpenAI's embedding model.
- Relevant passages are pulled from the index and included in the prompt, along with each source's title and URL.
- Users can also upload their own documents (PDF or TXT) to expand the knowledge base.
- A dropdown lets users choose which chunk of the uploaded file to include in the prompt, helping fine-tune the context.

This implementation demonstrates a strong RAG setup using OpenAI and FAISS, reinforcing both LO1 and LO2.

---

## Section 6: Additional Tools / Innovation

The system introduces several thoughtful enhancements that go beyond core functionality:

- Session memory: remembers the last two Q&A pairs to maintain conversational flow.
- Export options: download session history in CSV or TXT format.
- Voice playback: toggle read-aloud mode for any response.
- Test suite: automated tests cover retrieval, chunking, memory, prompting, and TTS logic.
- API key validation: real-time feedback for key entry, with subtle fading for success/failure messages.

These features reflect not just solid technical implementation, but also creativity, attention to user experience, and thoughtful problem-solving (LO2).

---

## Section 7: Code Quality & Modular Design

The codebase emphasizes clarity and maintainability through modular files and meaningful comments:

- Each component (`generator.py`, `retrieval.py`, `upload_utils.py`, etc.) is modular and isolated.
- API keys are handled securely via runtime entry.
- Tests for all critical logic are included under `tests/`.
- Readable prompts and internal logic are well-commented.
- Current test coverage is approximately 65%, with all critical logic covered.

The project follows version control conventions, uses Streamlit state management cleanly, and is documented through a [README.md](https://github.com/mohammed0823/ai-study-buddy/blob/master/README.md) and `Project.md` (this file).

---

## Conclusion

AI Study Buddy demonstrates a complete application of course concepts in artificial intelligence, including retrieval-augmented generation, prompt engineering, session memory, and modular design. Its well-designed interface and robust backend reflect a strong grasp of both theoretical AI methods and practical system development.

---