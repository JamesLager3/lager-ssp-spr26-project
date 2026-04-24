import os
import tempfile
import torch
from PyPDF2 import PdfReader
from unittest.mock import MagicMock, patch, mock_open
import json
from pathlib import Path
import pandas as pd

import extractor
import comparator
import executor

# ------------------------------------------------------------------ #
# Helpers shared by multiple tests                                      #
# ------------------------------------------------------------------ #

def _make_encoded(seq_len: int = 10):
    """Return a fake apply_chat_template dict for a sequence of `seq_len` tokens."""
    return {
        "input_ids":      torch.ones(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }


def _make_mock_tokenizer(seq_len: int = 10):
    tok = MagicMock()
    tok.apply_chat_template.return_value = _make_encoded(seq_len)
    tok.decode.return_value = "decoded output"
    tok.pad_token_id = 0          # <-- must be a real int, not a Mock
    tok.eos_token_id = 2          # <-- same
    tok.padding_side = "left"     # <-- if extractor checks this
    return tok


def _make_mock_model(new_tokens: int = 5):
    """Return a mock model whose generate() appends `new_tokens` zero-columns."""
    model = MagicMock()
    model.device = "cpu"

    def fake_generate(input_ids, attention_mask, **kwargs):
        batch, padded_len = input_ids.shape
        total = padded_len + new_tokens
        return torch.zeros(batch, total, dtype=torch.long)

    model.generate.side_effect = fake_generate
    return model


# ------------------------------------------------------------------ #
# extractor.py                                                           #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# load_pdf                                                             #
# ------------------------------------------------------------------ #

def test_load_pdf_valid():
    """Concatenates text from every page and returns the full string."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "page text "

    with patch("os.path.exists", return_value=True), \
        patch(f"extractor.PdfReader") as MockReader:
        MockReader.return_value.pages = [mock_page, mock_page]
        result = extractor.load_pdf("any.pdf")

    assert result == "page text page text ", f"Unexpected result: {result!r}"


def test_load_pdf_none_page():
    """Pages that return None from extract_text() are treated as empty strings."""
    none_page = MagicMock()
    none_page.extract_text.return_value = None
    text_page = MagicMock()
    text_page.extract_text.return_value = "real text"

    with patch("os.path.exists", return_value=True), \
         patch(f"extractor.PdfReader") as MockReader:
        MockReader.return_value.pages = [none_page, text_page]
        result = extractor.load_pdf("any.pdf")

    assert result == "real text", f"Unexpected result: {result!r}"


def test_load_pdf_not_found():
    """Raises FileNotFoundError when the path does not exist."""
    with patch("os.path.exists", return_value=False):
        try:
            extractor.load_pdf("missing.pdf")
            raise AssertionError("Expected FileNotFoundError was not raised")
        except FileNotFoundError:
            pass


def test_load_pdf_read_error():
    """Wraps reader exceptions in RuntimeError."""
    with patch("os.path.exists", return_value=True), \
         patch(f"extractor.PdfReader", side_effect=Exception("corrupt file")):
        try:
            extractor.load_pdf("bad.pdf")
            raise AssertionError("Expected RuntimeError was not raised")
        except RuntimeError:
            pass


# ------------------------------------------------------------------ #
# Prompt functions                                                      #
# ------------------------------------------------------------------ #

def _assert_prompt_contract(encoded, prompt, chunk_text: str, tok):
    """Shared assertions for all three prompt-building functions."""
    # Returns the right types
    assert isinstance(prompt, str),  "prompt must be a str"
    assert isinstance(encoded, dict), "encoded must be a dict"

    # Encoded dict has the expected keys
    assert "input_ids"      in encoded, "encoded missing 'input_ids'"
    assert "attention_mask" in encoded, "encoded missing 'attention_mask'"

    # apply_chat_template was called exactly once
    tok.apply_chat_template.assert_called_once()

    # The chunk text was passed somewhere inside the messages argument
    call_args = tok.apply_chat_template.call_args
    messages  = call_args[0][0]          # first positional arg
    all_content = " ".join(m["content"] for m in messages)
    assert chunk_text in all_content, "chunk text not forwarded into messages"

    # Prompt string is non-empty and contains task-level instructions
    assert len(prompt.strip()) > 0,        "prompt string is empty"
    assert "TASK"       in prompt,         "prompt missing TASK section"
    assert "SCHEMA"     in prompt,         "prompt missing SCHEMA TEMPLATE"
    assert "requirements" in prompt,       "prompt missing requirements key"


def test_zero_shot_prompt():
    """zero_shot_prompt returns a valid (encoded, prompt) tuple."""
    tok        = _make_mock_tokenizer()
    chunk_text = "1.1.1 Ensure auditing is configured."
    encoded, prompt = extractor.zero_shot_prompt(chunk_text, tok)
    _assert_prompt_contract(encoded, prompt, chunk_text, tok)


def test_few_shot_prompt():
    """few_shot_prompt returns a valid (encoded, prompt) tuple with an example."""
    tok        = _make_mock_tokenizer()
    chunk_text = "5.5.1 Manage Kubernetes RBAC users with AWS IAM Authenticator for Kubernetes (Manual)"
    encoded, prompt = extractor.few_shot_prompt(chunk_text, tok)
    _assert_prompt_contract(encoded, prompt, chunk_text, tok)

    # few-shot specific: the example shot should appear in the user message
    call_args   = tok.apply_chat_template.call_args
    messages    = call_args[0][0]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "aws iam authenticator" in user_content.lower(), \
        "few-shot example not present in user message"


def test_cot_prompt():
    """cot_prompt returns a valid (encoded, prompt) tuple with a PROCESS section."""
    tok        = _make_mock_tokenizer()
    chunk_text = "3.5.2 Ensure kernel parameters are hardened."
    encoded, prompt = extractor.cot_prompt(chunk_text, tok)
    _assert_prompt_contract(encoded, prompt, chunk_text, tok)

    assert "PROCESS" in prompt, "CoT prompt missing PROCESS section"


# ------------------------------------------------------------------ #
# generate_output                                                       #
# ------------------------------------------------------------------ #

def test_generate_output_single():
    """A single encoded dict returns a one-element list."""
    tok   = _make_mock_tokenizer(seq_len=10)
    model = _make_mock_model(new_tokens=5)

    results = extractor.generate_output(_make_encoded(10), tok, model)

    assert isinstance(results, list),  "result must be a list"
    assert len(results) == 1,          "single input must yield one output"
    assert isinstance(results[0], str), "each output must be a str"
    model.generate.assert_called_once()


def test_generate_output_batch():
    """A list of N encoded dicts yields N outputs in one generate() call."""
    tok   = _make_mock_tokenizer()
    model = _make_mock_model(new_tokens=5)

    # Deliberately use different sequence lengths to exercise left-padding
    batch = [_make_encoded(6), _make_encoded(10), _make_encoded(8)]
    results = extractor.generate_output(batch, tok, model)

    assert isinstance(results, list), "result must be a list"
    assert len(results) == 3,         "batch of 3 must yield 3 outputs"
    assert all(isinstance(r, str) for r in results), "every output must be a str"
    # Crucially, the model is called only once for the whole batch
    model.generate.assert_called_once()


# ------------------------------------------------------------------ #
# log_results                                                           #
# ------------------------------------------------------------------ #

def test_log_results_content():
    """All four fields are written to the log file."""
    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
        tmp_path = f.name
    try:
        extractor.log_results("test-model", "my prompt", "Zero-Shot", "my output", file=tmp_path)
        content = open(tmp_path).read()
        assert "test-model" in content, "LLM name missing from log"
        assert "my prompt"  in content, "prompt text missing from log"
        assert "Zero-Shot"  in content, "prompt type missing from log"
        assert "my output"  in content, "LLM output missing from log"
    finally:
        os.remove(tmp_path)


def test_log_results_appends():
    """Calling log_results twice appends both entries to the same file."""
    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
        tmp_path = f.name
    try:
        extractor.log_results("model-a", "prompt a", "Zero-Shot", "output a", file=tmp_path)
        extractor.log_results("model-b", "prompt b", "Few-Shot",  "output b", file=tmp_path)
        content = open(tmp_path).read()
        assert "model-a"  in content and "model-b"  in content, "both entries must be present"
        assert "output a" in content and "output b" in content, "both outputs must be present"
        assert content.count("=" * 50) == 2, "expected two separator lines"
    finally:
        os.remove(tmp_path)


# ------------------------------------------------------------------ #
# comparator.py tests                                                #
# ------------------------------------------------------------------ #

def test_load_yaml_valid():
    """Verifies that YAML content is correctly loaded into a dict."""
    content = "key: value\nlist: [1, 2]"
    with patch("builtins.open", MagicMock(return_value=tempfile.TemporaryFile())):
        with patch("yaml.safe_load", return_value={"key": "value"}):
            result = comparator._load_yaml("fake.yaml")
            assert result == {"key": "value"}

def test_extract_elements_logic():
    """Tests extraction of KDE names and requirements from raw dict data."""
    data = {
        "block1": {
            "name": "ElementA",
            "requirements": {
                "req1": "req_val_1"
            }
        },
        "block2": {
            "name": "ElementB",
            "requirements": {
                "req1": "req_val_2"
            }
        }
    }
    elements = comparator._extract_elements(data)
    assert elements["ElementA"] == {"req_val_1"}
    assert elements["ElementB"] == {"req_val_2"}

def test_compare_element_names_diff():
    """Verifies output file content when names differ between files."""
    # Ensure return values are dicts of sets to support subtraction logic
    e1 = {"A": set()}
    e2 = {"B": set()}
    
    with patch("comparator._extract_elements", side_effect=[e1, e2]), \
         patch("builtins.open", mock_open()) as m:
        
        # Pass a string for the output_path as defined in the function signature
        comparator.compare_element_names("f1.yaml", "f2.yaml", "out.txt")
        
        # Updated to expect the string 'out.txt' to match the actual call
        m.assert_called_with("out.txt", "w", encoding="utf-8")
        
        written_data = "".join(call.args[0] for call in m().write.call_args_list)
        assert "A" in written_data
        assert "B" in written_data


def test_compare_element_requirements_diff():
    """Verifies CSV-style output when requirement values differ."""
    # Logic: reqs1 - reqs2 for common KDE names.
    e1 = {"KDE": {"req1"}}
    e2 = {"KDE": {"req2"}}

    with patch("comparator._extract_elements", side_effect=[e1, e2]), \
         patch("builtins.open", mock_open()) as m:
        
        comparator.compare_element_requirements("f1.yaml", "f2.yaml", "out.txt")
        
        written_data = "".join(call.args[0] for call in m().write.call_args_list)
        # Verify the custom CSV-like format used in comparator.py
        assert "PRESENT-IN-f1.yaml,req1" in written_data
        assert "PRESENT-IN-f2.yaml,req2" in written_data

# ------------------------------------------------------------------ #
# executor.py tests                                                  #
# ------------------------------------------------------------------ #

def test_read_input_files():
    """Ensures differences are parsed correctly from name and req diff files."""
    name_content = "KDE_ONLY_IN_1\n"
    # Note: executor.py splits by "," and expects 4 parts
    req_content = "KDE_COMMON,ABSENT-IN-f2,PRESENT-IN-f1,SpecificReq\n"
    
    # We use side_effect to provide different content for the two open() calls
    with patch("builtins.open", side_effect=[
        mock_open(read_data=name_content).return_value,
        mock_open(read_data=req_content).return_value
    ]):
        diffs = executor.read_input_files("n.txt", "r.txt")
        assert "KDE_ONLY_IN_1" in diffs
        assert "KDE_COMMON:SpecificReq" in diffs

def test_generate_control_file_mapping():
    """Tests that keywords in differences map to correct Control IDs."""
    # We mock read_input_files so we don't have to worry about file I/O here
    with patch("executor.read_input_files", return_value=["privileged container"]), \
         patch("builtins.open", mock_open()) as m:
        
        executor.generate_control_file("n.txt", "r.txt", "out.txt")
        
        # Collect what was written to the file
        written_data = "".join(call.args[0] for call in m().write.call_args_list)
        # 'privileged' maps to C-0057 in the mapping
        assert "C-0057" in written_data


@patch("executor.subprocess.run")
@patch("executor.shutil.which", return_value="/usr/bin/kubescape")
def test_run_kubescape_success(mock_which, mock_run):
    """Verifies that Kubescape CLI is called and output is parsed into a DataFrame."""
    # Mock successful Kubescape JSON output
    mock_run.return_value = MagicMock(
        stdout=json.dumps({
            "summaryDetails": {"controls": {"C-0001": {"severity": "high", "name": "test-ctrl", "ResourceCounters": {"failedResources": 1}}}},
            "results": [{"resourceID": "pod.yaml", "controls": [{"controlID": "C-0001", "status": {"status": "failed"}}]}]
        }),
        stderr=""
    )
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", MagicMock(return_value=MagicMock(read=lambda: "C-0001"))), \
         patch("executor.zipfile.ZipFile"), \
         patch("executor.tempfile.mkdtemp", return_value="/tmp/ks"):
        
        df = executor.run_kubescape("controls.txt", "archive.zip")
        assert not df.empty
        assert df.iloc[0]["Control name"] == "test-ctrl"
        assert df.iloc[0]["Severity"] == "high"


def test_generate_csv():
    """Checks if DataFrame is correctly exported to CSV."""
    df = pd.DataFrame([{
        "FilePath": "test.yaml", 
        "Severity": "Medium", 
        "Control name": "Rule1", 
        "Failed resources": 1, 
        "All Resources": 1, 
        "Compliance score": 0
    }])
    
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        executor.generate_csv(df, "test.csv")
        mock_to_csv.assert_called_once_with("test.csv", index=False)

# ------------------------------------------------------------------ #
# Test suite                                                            #
# ------------------------------------------------------------------ #

def run_test_suite():
    extractor_functions = [
        # load_pdf
        test_load_pdf_valid,
        test_load_pdf_none_page,
        test_load_pdf_not_found,
        test_load_pdf_read_error,
        # prompt functions
        test_zero_shot_prompt,
        test_few_shot_prompt,
        test_cot_prompt,
        # generate_output
        test_generate_output_single,
        test_generate_output_batch,
        # log_results
        test_log_results_content,
        test_log_results_appends,
    ]

    comparator_functions = [
        test_load_yaml_valid,
        test_extract_elements_logic,
        test_compare_element_names_diff,
        test_compare_element_requirements_diff
    ]

    executor_functions = [
        test_read_input_files,
        test_generate_control_file_mapping,
        test_run_kubescape_success,
        test_generate_csv
    ]

    file_functions = [
        (extractor_functions, "extractor"), 
        (comparator_functions, "comparator"), 
        (executor_functions, "executor")
    ]

    for functions, name in file_functions:
        passed = 0
        failed = 0
        for func in functions:
            try:
                func()
                print(f"  PASS  {func.__name__}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {func.__name__}: {exc}")
                failed += 1
        print(f"{name}: {passed} passed, {failed} failed.\n")

if __name__ == "__main__":
    run_test_suite()
    if os.path.exists('test.csv'):
        os.remove('test.csv')