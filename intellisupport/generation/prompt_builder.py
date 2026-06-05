import logging
from retrieval.vector_store import RetrievedChunk
from classification.intent_classifier import IntentResult

logger = logging.getLogger(__name__)

_TONE_BY_INTENT = {
    "billing": (
        "Be precise about pricing and plan details. "
        "Never speculate about costs or future pricing changes."
    ),
    "technical_issue": (
        "Provide clear, numbered troubleshooting steps. "
        "Be systematic and actionable."
    ),
    "feature_request": (
        "Describe existing capabilities accurately. "
        "Do not promise features that are not documented."
    ),
    "integration": (
        "Be specific about required configuration steps, "
        "permissions, and credential requirements."
    ),
    "account_management": (
        "Be security-conscious and precise. "
        "Guide the user through each step clearly."
    ),
    "data_and_export": (
        "Be clear about supported formats, retention policies, "
        "and any data limitations."
    ),
    "general_inquiry": "Be concise and helpful.",
}


class PromptBuilder:

    def build_rag_prompt(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        intent: IntentResult,
    ) -> list[dict]:
        tone = _TONE_BY_INTENT.get(intent.intent, _TONE_BY_INTENT["general_inquiry"])

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[CHUNK {i} | doc_id: {chunk.doc_id} | chunk_id: {chunk.chunk_id}]\n"
                f"{chunk.content}"
            )
        context_block = "\n\n".join(context_parts)

        system_content = (
            "You are Nexora's AI Support Assistant.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer ONLY using information explicitly present in the context below.\n"
            "2. If the context does not contain sufficient information to answer the question, "
            'respond with exactly: "I don\'t have information about that in the current '
            'documentation. Please contact support@nexora.io for further assistance."\n'
            "3. Never invent, guess, or infer facts that are not directly stated in the context.\n"
            "4. End your response by citing the chunk_ids you used, formatted as: "
            "[Sources: chunk_doc_XXX_N, chunk_doc_YYY_N]\n\n"
            f"Detected customer intent: {intent.intent} (confidence: {intent.confidence:.2f})\n"
            f"Tone guidance: {tone}"
        )

        user_content = (
            f"CONTEXT:\n{context_block}\n\n"
            f"CUSTOMER QUESTION: {query}\n\n"
            "Provide a clear, accurate answer based strictly on the context. "
            "Cite all chunk_ids you referenced at the end."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def build_clarification_prompt(
        self, query: str, intent: IntentResult
    ) -> list[dict]:
        system_content = (
            "You are Nexora's AI Support Assistant. "
            "No relevant documentation was found for this question. "
            "Politely acknowledge that you do not have specific information available "
            "and ask one focused clarifying question to better understand the customer's needs. "
            "Do not fabricate or guess any information."
        )
        user_content = (
            f"Customer question: {query}\n"
            f"Detected intent: {intent.intent}\n\n"
            "Acknowledge the documentation gap and ask a single clarifying question."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def estimate_prompt_tokens(self, messages: list[dict]) -> int:
        return int(sum(len(m["content"].split()) * 1.3 for m in messages))
