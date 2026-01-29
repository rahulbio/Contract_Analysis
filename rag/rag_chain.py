# rag/rag_chain.py

def build_rag_engine(llm):
    """
    One-time setup: prompts, retriever, templates
    """
    return {
        "llm": llm
    }


def run_rag_reasoning(rag_engine, rag_context, question):
    """
    Per-question reasoning
    """
    prompt = f"""
You are explaining a contract to a layperson.

KNOWN FACTS:
{rag_context}

QUESTION:
{question}

RULES:
- Use ONLY known facts
- Be neutral
- Explain simply
"""

    return rag_engine["llm"](prompt)