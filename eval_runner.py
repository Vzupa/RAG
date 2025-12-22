
import requests
import csv
import time
import os
from typing import List

# --- Configuration ---
API_BASE = "http://127.0.0.1:8000"
QUESTIONS: List[str] = [
    "What learning paradigm is proposed by Mbona and Eloff (2022) for detecting zero-day intrusions?",
    "On which type of data is the model in Mbona and Eloff (2022) primarily trained?",
    "Which principle is used for feature selection in Mbona and Eloff (2022)?",
    "Which machine learning model achieved the best performance in Mbona and Eloff (2022)?",
    "Why are supervised learning methods weak for zero-day attack detection according to the literature?",
    "What types of threats are addressed in Wategaonkar et al. (2024)?",
    "What analysis technique is combined with machine learning in Wategaonkar et al. (2024)?",
    "What is the primary impact of the React2Shell vulnerability?",
    "Which JavaScript frameworks are affected by the React2Shell vulnerability?",
    "Why is anomaly-based detection suitable for zero-day attacks?",
    "What exact CVSS score is assigned to the React2Shell vulnerability?",
    "Which specific line of source code causes the React2Shell vulnerability?",
    "What GPU model was used to train the models in Mbona and Eloff (2022)?",
    "What was the exact training time in seconds for the OCSVM model?",
    "Which future version of React fully fixes the React2Shell vulnerability?",
]

EXPECTED_ANSWERS: List[str] = [
    "Semi-supervised machine learning",
    "Benign network traffic only",
    "Benford’s Law",
    "One-Class Support Vector Machine (OCSVM)",
    "They require labeled attack data and generalize poorly to unseen attacks",
    "Insider threats and zero-day vulnerabilities",
    "Behavioral analysis",
    "Remote code execution",
    "React and Next.js",
    "Zero-day attacks deviate from normal behavior",
    "Not mentioned in the documents",
    "Not specified in the documents",
    "No GPU usage is mentioned",
    "Not reported in the documents",
    "Not mentioned in the documents",
]

# --- Main evaluation logic ---

def run_evaluation() -> None:
    """
    Runs a batch evaluation of questions against the API in different modes
    and logs the results to a CSV file.
    """
    top_k = 4
    output_filename = f"eval_results_{top_k}.csv"

    # Define modes for evaluation
    modes = {
        "LLM": {"endpoint": "/llm/ask", "payload": {"multiquery": False, "hyde": False}},
        "RAG": {"endpoint": "/rag/ask", "payload": {"top_k": top_k, "multiquery": False, "hyde": False}},
        "RAG_MQ": {"endpoint": "/rag/ask", "payload": {"top_k": top_k, "multiquery": True, "hyde": False}},
        "RAG_HyDE": {"endpoint": "/rag/ask", "payload": {"top_k": top_k, "multiquery": False, "hyde": True}},
        "RAG_MQ_HyDE": {"endpoint": "/rag/ask", "payload": {"top_k": top_k, "multiquery": True, "hyde": True}},
    }

    # Prepare CSV file
    file_exists = os.path.exists(output_filename)
    with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "question",
            "mode",
            "multiquery",
            "hyde",
            "client_latency_seconds",
            "backend_latency_seconds",
            "llm_latency_seconds",
            "answer",
            "expected_answer",
            "correctness",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run evaluation for each question and mode
        for i, question in enumerate(QUESTIONS):
            for mode_name, config in modes.items():

                payload = {"question": question, **config["payload"]}
                endpoint = config["endpoint"]

                try:
                    # Measure client-side latency
                    start_time_client = time.perf_counter()
                    response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
                    end_time_client = time.perf_counter()

                    client_latency = end_time_client - start_time_client
                    response.raise_for_status()
                    data = response.json()

                    # Prepare row for CSV
                    backend_latency = data.get("backend_latency")
                    llm_latency = data.get("llm_latency")

                    row = {
                        "question": question,
                        "mode": mode_name,
                        "multiquery": payload.get("multiquery"),
                        "hyde": payload.get("hyde"),
                        "client_latency_seconds": f"{client_latency:.4f}",
                        "backend_latency_seconds": f"{backend_latency:.4f}" if backend_latency is not None else "",
                        "llm_latency_seconds": f"{llm_latency:.4f}" if llm_latency is not None else "",
                        "answer": data.get("answer", "N/A"),
                        "expected_answer": EXPECTED_ANSWERS[i],
                        "correctness": "",
                    }
                    writer.writerow(row)

                    print(f"[{mode_name}] Q{i+1:02d} – {client_latency:.3f}s")

                except requests.exceptions.RequestException as e:
                    print(f"Error calling API for Q{i+1} in mode {mode_name}: {e}")
                    row = {
                        "question": question,
                        "mode": mode_name,
                        "multiquery": payload.get("multiquery"),
                        "hyde": payload.get("hyde"),
                        "answer": f"ERROR: {e}",
                        "expected_answer": EXPECTED_ANSWERS[i],
                        "correctness": "ERROR",
                    }
                    writer.writerow(row)

    print(f"\nEvaluation complete. Results saved to '{output_filename}'.")


if __name__ == "__main__":
    run_evaluation()
