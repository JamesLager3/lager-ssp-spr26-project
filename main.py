import sys
import os
import zipfile
import extractor
import comparator
import executor

if __name__ == "__main__":
    if os.path.exists('llm_outputs.txt'):
        os.remove('llm_outputs.txt')
    if os.path.exists('requirement_differences.txt'):
        os.remove('requirement_differences.txt')
    if os.path.exists('name_differences.txt'):
        os.remove('name_differences.txt')
    if os.path.exists('controls.txt'):
        os.remove('controls.txt')

    doc1_path, doc2_path = sys.argv[1], sys.argv[2]
    doc1_yaml = doc1_path[0:-4] + "-kdes.yaml"
    doc2_yaml = doc2_path[0:-4] + "-kdes.yaml"

    extractor.main(doc1_path, doc2_path)

    comparator.main(doc1_yaml, doc2_yaml)

    executor.main("name_differences.txt", "requirement_differences.txt")

    if os.path.exists('kubescape_output.json'):
        os.remove('kubescape_output.json')