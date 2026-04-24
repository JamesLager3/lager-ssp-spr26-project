import sys
import os
import zipfile
import extractor
import comparator
import executor

if __name__ == "__main__":
    doc1_path, doc2_path = sys.argv[1], sys.argv[2]
    doc1_yaml = doc1_path[0:-4] + "-kdes.yaml"
    doc2_yaml = doc2_path[0:-4] + "-kdes.yaml"

    extractor.main(doc1_path, doc2_path)

    comparator.main(doc1_yaml, doc2_yaml)

    executor.main("name_differences.txt", "requirement_differences.txt")

    if os.path.exists('kubescape_output.json'):
        os.remove('kubescape_output.json')