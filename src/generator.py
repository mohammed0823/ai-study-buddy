import streamlit as st
from openai import OpenAI

# Builds a complete prompt using context, memory, and style settings
def build_prompt(question, docs_metadata, style="Default", memory_block="", cot=False):
    # Compile the document chunks into a referenceable context block
    context = ""
    for i, (chunk, meta) in enumerate(docs_metadata, start=1):
        context += f"\n[Source {i}] Title: {meta['title']}\n{chunk}\nURL: {meta['url']}\n"

    # Define available prompt styles the user can choose from
    instructions = {
        "Default": "Answer the question using the context provided.",
        "Concise": "Answer briefly and directly using only relevant context.",
        "Beginner-Friendly": "Explain clearly as if to someone new to AI. Use simple language.",
        "Explain Step-by-Step": "Answer the question by explaining your reasoning step by step.",
        "With Citations Only": "Answer using only the information provided in the sources below. Cite them clearly throughout your response."
    }

    # Select the user's chosen style or fallback to Default
    style_instruction = instructions.get(style, instructions["Default"])

    # If CoT is enabled, encourage step-by-step reasoning
    cot_instruction = "\nRespond by thinking through the answer step by step." if cot else ""

    # Assemble the final prompt for the model
    prompt = (
        "You are an expert AI assistant helping users understand concepts in artificial intelligence.\n\n"
        f"{style_instruction}{cot_instruction}\n"
        f"{memory_block}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    return prompt

# Sends the prompt to OpenAI and returns the generated answer
def generate_answer(prompt, temperature=0.2, max_tokens=300):
    client = OpenAI(api_key=st.session_state.get("openai_api_key"))
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Answer generation failed: {e}")
        return "I'm sorry, I couldn't generate an answer right now."