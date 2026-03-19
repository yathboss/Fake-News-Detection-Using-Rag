from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.embed_index import DEFAULT_EMBED_MODEL, build_or_load_faiss_index
from utils.llm_infer import generate_verdict
from utils.ner_kg import build_basic_kg
from utils.retriever import FaissRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpus"
INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"


class ClaimVerificationPipeline:
    def __init__(self, embed_model_name: str = DEFAULT_EMBED_MODEL):
        index, metadata, embedder = build_or_load_faiss_index(
            corpus_dir=CORPUS_DIR,
            index_dir=INDEX_DIR,
            model_name=embed_model_name,
            force_rebuild=False,
        )
        self.retriever = FaissRetriever(index=index, metadata=metadata, embedder=embedder)

    def verify_claim(self, claim: str, top_k: int = 3) -> Dict:
        retrieved = self.retriever.search(claim, top_k=top_k)
        llm_ready_evidence: List[dict] = []
        for hit in retrieved:
            llm_ready_evidence.append(
                {
                    "rank": hit["rank"],
                    "doc_id": hit["doc_id"],
                    "title": hit["title"],
                    "source": hit["source"],
                    "chunk_id": hit["chunk_id"],
                    "text": hit["text"],
                    "score": round(hit["score"], 4),
                }
            )

        entities, kg_triples, kg_summary = build_basic_kg(
            claim=claim,
            evidence_texts=[item["text"] for item in llm_ready_evidence],
        )
        verdict = generate_verdict(
            claim=claim,
            evidence=llm_ready_evidence,
            entities=entities,
            kg_triples=kg_triples,
        )

        return {
            "claim": claim,
            "predicted_label": verdict["predicted_label"],
            "explanation": verdict["explanation"],
            "confidence": max(0.0, min(float(verdict["confidence"]), 1.0)),
            "evidence": llm_ready_evidence,
            "entities": entities,
            "kg_triples": kg_triples,
            "retrieval_summary": (
                f"Retrieved {len(llm_ready_evidence)} evidence chunks from local corpus. "
                f"KG nodes: {len(kg_summary['nodes'])}, edges: {kg_summary['edges']}."
            ),
            "model_used": verdict["model_used"],
        }
