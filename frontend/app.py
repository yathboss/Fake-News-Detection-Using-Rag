from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("Fake News Detection using KG + RAG + Local Llama")
st.caption("Phase 1 prototype: claim -> evidence retrieval -> KG context -> local verdict")

default_claim = "Climate change is a hoax invented by scientists."
claim = st.text_area("Enter a claim", value=default_claim, height=140)
top_k = st.slider("Number of evidence snippets", min_value=1, max_value=5, value=3)

if st.button("Verify Claim", type="primary", use_container_width=True):
    if not claim.strip():
        st.warning("Please enter a claim first.")
    else:
        with st.spinner("Retrieving evidence and asking the local model..."):
            response = requests.post(
                f"{API_BASE_URL}/verify",
                json={"claim": claim.strip(), "top_k": top_k},
                timeout=180,
            )

        if response.ok:
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Label", data["predicted_label"])
            col2.metric("Confidence", f"{data['confidence']:.2f}")
            col3.metric("Model Used", data["model_used"])

            st.subheader("Explanation")
            st.write(data["explanation"])

            st.subheader("Top Evidence")
            for item in data["evidence"]:
                with st.container(border=True):
                    st.markdown(
                        f"**#{item['rank']} {item['title']}**  \n"
                        f"`doc_id`: {item['doc_id']} | `chunk_id`: {item['chunk_id']} | "
                        f"`source`: {item['source']} | `score`: {item['score']:.4f}"
                    )
                    st.write(item["text"])

            st.subheader("Entities")
            if data["entities"]:
                st.write(", ".join(data["entities"]))
            else:
                st.write("No entities extracted.")

            st.subheader("Basic KG Context")
            if data["kg_triples"]:
                for triple in data["kg_triples"][:10]:
                    st.code(f"({triple['subject']}) -[{triple['relation']}]-> ({triple['object']})")
            else:
                st.write("No KG triples generated.")

            if data.get("retrieval_summary"):
                st.info(data["retrieval_summary"])
        else:
            st.error(f"Backend error: {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)
