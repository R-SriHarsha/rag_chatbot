from search_pipeline.retriever import get_retriever
from search_pipeline.llm import get_llm


def build_rag_chain():

    retriever = get_retriever()
    llm = get_llm()
    

    def ask(question, allow_general=False):

        docs = retriever.vectorstore.similarity_search_with_score(question, k=5)

        context = []
        scores = []

        for doc, score in docs:
            context.append(doc.page_content.strip())
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        keyword_match = question.lower() in " ".join(context).lower()

        relevant = avg_score < 1.2 or keyword_match

        # --- Outside document ---
        if not relevant and not allow_general:
            return {
                "answer": "Question may be outside the uploaded documents.",
                "context": context,
                "needs_confirmation": True
            }

        if not relevant and allow_general:
            resp = llm.invoke(question)
            answer = resp.content if hasattr(resp, "content") else str(resp)
            return {
                "answer": answer,
                "context": context,
                "needs_confirmation": False
            }

        prompt = f"""
You are an AI Teacher Assistant.

RULES:
- Use ONLY the given context
- If context is insufficient say:
  "The document does not provide sufficient information."
- Give definition + explanation (5–7 sentences)
- Add bullet key points

Context:
{" ".join(context)}

Question:
{question}

Answer:
"""

        resp = llm.invoke(prompt)
        answer = resp.content if hasattr(resp, "content") else str(resp)

        if "The document does not provide sufficient information." in answer and not allow_general:
            return {
                "answer": "Not enough info in documents.",
                "context": context,
                "needs_confirmation": True
            }

        return {
            "answer": answer,
            "context": context,
            "needs_confirmation": False
        }

    return ask
