import os
import sys
sys.path.append(os.getcwd())

from rag_pipeline import OdishaRAGPipeline

def run_tests():
    print("Initializing RAG Pipeline...")
    pipeline = OdishaRAGPipeline()
    
    print("\n--- TEST 1: Retrieve context for 'aryapalli' ---")
    context, citations = pipeline.retrieve_context("aryapalli")
    print(f"Retrieved {len(citations)} citations.")
    for idx, c in enumerate(citations):
        print(f"[{idx+1}] File: {c['filename']}, Excerpt: {c['content'][:150]}...")
    
    # Verify that we matched the Ganjam tourist spot list containing Aryapalli
    has_aryapalli = any("aryapalli" in c["content"].lower() for c in citations)
    if has_aryapalli:
        print("[SUCCESS] Found Aryapalli in retrieved context chunks!")
    else:
        print("[FAILURE] Aryapalli not found in retrieved context!")
        
    print("\n--- TEST 2: Run LLM query for 'aryapalli' ---")
    ans1 = pipeline.query_llm_with_context("aryapalli", context)
    print("Answer:\n", ans1)
    
    print("\n--- TEST 3: Retrieve context with history (Follow-up: 'where is it') ---")
    history = [
        {"role": "user", "content": "aryapalli"},
        {"role": "assistant", "content": ans1},
        {"role": "user", "content": "where is it"}
    ]
    
    condensed_q = pipeline.condense_query("where is it", history)
    print("Condensed follow-up query:", condensed_q)
    
    context2, citations2 = pipeline.retrieve_context("where is it", history=history)
    print(f"Retrieved {len(citations2)} citations for follow-up.")
    has_aryapalli2 = any("aryapalli" in c["content"].lower() for c in citations2)
    if has_aryapalli2:
        print("[SUCCESS] Follow-up query retrieved Aryapalli context successfully!")
    else:
        print("[FAILURE] Follow-up query failed to retrieve Aryapalli context!")
        
    print("\n--- TEST 4: Query LLM for follow-up 'where is it' with history ---")
    ans2 = pipeline.query_llm_with_context("where is it", context2, history=history)
    print("Answer:\n", ans2)
    
    # Check if the answer correctly mentions Ganjam district
    if "ganjam" in ans2.lower():
        print("[SUCCESS] Answer correctly refers to Ganjam district!")
    else:
        print("[FAILURE] Answer does not refer to Ganjam district.")

if __name__ == "__main__":
    run_tests()
