import os
import yaml
from typing import Tuple, Dict, List
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import re

torch.cuda.empty_cache()

def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")


def zero_shot_prompt(text: str, tokenizer) -> str:
    prompt = """
        TASK: You are a security engineer that needs to detect requirements for specific data elements. 
        Identify Key Data Elements in security text, and output into a FLAT JSON SCHEMA.

        CRITICAL RULES:
        1. Look for pieces data that have security or audit requirements.
        2. A data element shouldnt begin with or contain verbs like: "Configure", "Ensure", "Enable", or "Allow". These are requirements.
        3. Use ONLY these keys: "element1", "element2", "name", "requirements", "reqX".
        4. The value of "requirements" MUST be a dictionary of "reqX" keys, not a list.
        5. Ignore lines of code, version numbers, page numbers. 
        
        SCHEMA TEMPLATE:
        {
            "element1": {
                "name": "Short Descriptive Title",
                "requirements": {
                    "req1": "A related requirement for the KDE"
                }
            }
        }
    """
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Document:\n<DOC_START>\n{text}\n<DOC_END>\n\nJSON Output:"}
    ]
    return tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ), prompt

def few_shot_prompt(text: str, tokenizer) -> str:
    example_shot = """
        Extract KDEs from text. Output ONLY JSON. 
        Example Input: '5.5.1 Manage Kubernetes RBAC users with AWS IAM Authenticator for Kubernetes (Manual)
        Profile Applicability:
        • Level 2
        Description:
        Amazon EKS uses IAM to provide authentication to your Kubernetes cluster through the
        AWS IAM Authenticator for Kubernetes. You can configure the stock kubectl client to
        work with Amazon EKS by installing the AWS IAM Authenticator for Kubernetes and
        modifying your kubectl configuration file to use it for authentication.'
        Example Output: 
        {
            "element1": {
                "name": "AWS IAM Authenticator",
                "requirements": {
                    "req1": "Configure the stock kubectl client to work with Amazon EKS by installing the AWS IAM Authenticator for Kubernetes"
                    "req2": "Modify your kubectl configuration file to use it for authentication."
                }
            }
        }
    """
    prompt = """
        TASK: You are a security engineer that needs to detect requirements for specific data elements. 
        Identify Key Data Elements in security text, and output into a FLAT JSON SCHEMA.

        CRITICAL RULES:
        1. Each element MUST:
        - be a noun (data entity)
        - have associated requirement actions
        2. The value of "requirements" MUST be a dictionary of "reqX" keys, not a list.
        3. Ignore lines of code, version numbers, page numbers.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"HERE IS AN EXAMPLE:{example_shot}\n\nYOUR TASK: Analyze the following:\n<TEXT>\n{text}\n</TEXT>\n\nOutput nested dict:"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ), prompt


def cot_prompt(text: str, tokenizer) -> str:
    prompt = """
        TASK:
        Find Key data elements in the document.

        PROCESS:
        1. Find DATA ELEMENTS with security or audit requirements.
        - These should be nouns, and keep them concise
        2. Group related secturity requirements.
        3. Generate JSON.
        - Reason internally, but NEVER output reasoning.
        - Use ONLY these keys: "element1", "element2", "name", "requirements", "reqX".
        - The value of "requirements" MUST be dictionary of "reqX" keys, not a list.

        SCHEMA TEMPLATE:
        {
            "element1": {
                "name": "Name of data element",
                "requirements": {
                    "req1": "A related requirement for the data"
                    "req2": "A related requirement for the data"
                }
            }
        }
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Document:\n<text>\n{text}\n</text>"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ), prompt


def generate_output(prompt_batch_or_list, tokenizer, model, max_new_tokens=2048) -> List[str]:
    if not isinstance(prompt_batch_or_list, list):
        prompt_batch_or_list = [prompt_batch_or_list]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Extract raw 1-D tensors from each {"input_ids": [1, L], "attention_mask": [1, L]}
    ids_list   = [enc["input_ids"].squeeze(0)      for enc in prompt_batch_or_list]
    masks_list = [enc["attention_mask"].squeeze(0) for enc in prompt_batch_or_list]

    # Left-pad all sequences to the length of the longest one
    max_len = max(t.shape[0] for t in ids_list)

    padded_ids, padded_masks = [], []
    for ids, mask in zip(ids_list, masks_list):
        pad_len = max_len - ids.shape[0]
        padded_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
        )
        padded_masks.append(
            torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
        )

    input_ids      = torch.stack(padded_ids).to(model.device)
    attention_mask = torch.stack(padded_masks).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # Greedy decoding is more stable for JSON
            repetition_penalty=1.2,   # Stops the "reprinting the same line" loop
            temperature=0.0,
        )

    # The model appended new tokens after position max_len for every sequence
    return [
        tokenizer.decode(out[max_len:], skip_special_tokens=True)
        for out in outputs
    ]


def log_results(llm_name: str, prompt: str, prompt_type: str, output: str, file="llm_outputs.txt"):
    with open(file, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"*LLM Name*\n{llm_name}\n")
        f.write(f"*Prompt Used*\n{prompt}\n")
        f.write(f"*Prompt Type*\n{prompt_type}\n")
        f.write(f"*LLM Output*\n{output}\n")


def main(doc1_path, doc2_path):
    doc_paths = [doc1_path, doc2_path]

    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    CHUNK_SIZE = 3000 
    OVERLAP    = 300
    BATCH_SIZE = 12

    methods = [
        ("Zero-Shot", zero_shot_prompt),
        ("Few-Shot",  few_shot_prompt),
        ("CoT",       cot_prompt),
    ]

    for doc_path in doc_paths:
        doc_name = os.path.basename(doc_path).replace(".pdf", "")
        text     = load_pdf(doc_path)
        tokens   = tokenizer.encode(text)

        doc_kdes = {}

        print(f"Processing {doc_name} ({len(tokens)} tokens)...")

        all_chunks = []
        for i in range(0, len(tokens), CHUNK_SIZE - OVERLAP):
            chunk_tokens = tokens[i : i + CHUNK_SIZE]
            all_chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

        for batch_start in range(0, len(all_chunks), BATCH_SIZE):
            batch_chunks = all_chunks[batch_start : batch_start + BATCH_SIZE]
            batch_end    = batch_start + len(batch_chunks)  # exclusive, for logging

            print(f"  [Chunks {batch_start + 1}-{batch_end}] Running 3-Prompt Extraction...")

            for method_name, prompt_func in methods:
                # Build one encoded prompt per chunk in the batch
                encoded_prompts = []
                prompt_texts    = []
                for chunk_text in batch_chunks:
                    full_prompt, prompt_text = prompt_func(chunk_text, tokenizer)
                    encoded_prompts.append(full_prompt)
                    prompt_texts.append(prompt_text)

                # Single generate() call covers the whole batch
                raw_outputs = generate_output(encoded_prompts, tokenizer, model)

                # Log and merge each output exactly as before
                for raw_output, prompt_text in zip(raw_outputs, prompt_texts):
                    log_results(
                        llm_name="gemma-3-1b-it",
                        prompt=prompt_text,
                        prompt_type=method_name,
                        output=raw_output,
                    )

                    clean = re.sub(r"```json", "", raw_output).strip()
                    start_idx = clean.find('{')
                    end_idx = clean.rfind('}')
                    if start_idx == -1 or end_idx == -1:
                        continue
                    
                    clean = clean[start_idx:end_idx+1]

                    try:
                        parsed = json.loads(clean)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(parsed, dict):
                        continue

                    for key, data in parsed.items():
                        if not isinstance(data, dict): continue
                        name = data.get("name")
                        reqs = data.get("requirements")

                        if not name or not isinstance(reqs, dict):
                            continue

                        valid_req_texts = [str(v).strip() for v in reqs.values() if str(v).strip()]
                        
                        if not valid_req_texts:
                            continue

                        name_key = name.strip()
                        if name_key not in doc_kdes:
                            doc_kdes[name_key] = set()
                        
                        for r_text in valid_req_texts:
                            doc_kdes[name_key].add(r_text)

            torch.cuda.empty_cache()

        # After all chunks are processed, finalize the "elementX" naming
        final_yaml_data = {}
        for i, (name, req_set) in enumerate(doc_kdes.items(), 1):
            req_dict = {f"req{j}": text for j, text in enumerate(sorted(req_set), 1)}
            final_yaml_data[f"element{i}"] = {
                "name": name,
                "requirements": req_dict
            }

        # Save cumulative result for the file
        with open(f"{doc_name}-kdes.yaml", "w") as f:
            yaml.dump(final_yaml_data, f, sort_keys=False)
        print(f"Completed {doc_name}. Saved YAML.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])