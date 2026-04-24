from pathlib import Path
import sys

import yaml

def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _extract_elements(data: dict) -> dict:
    elements = {}

    for block_key, block_val in data.items():
        if not isinstance(block_val, dict):
            continue

        kde_name = str(block_val.get("name", block_key)).strip()

        reqs_raw = block_val.get("requirements") or {}
        if isinstance(reqs_raw, dict):
            # req1: "...", req2: "..." - only values matter
            req_values = {
                str(v).strip()
                for v in reqs_raw.values()
                if v is not None
            }
        elif isinstance(reqs_raw, list):
            # Tolerate a plain list as well
            req_values = {str(v).strip() for v in reqs_raw if v is not None}
        else:
            req_values = set()

        if kde_name in elements:
            raise ValueError(
                f"Duplicate KDE name '{kde_name}' found in the YAML. "
                "Each element block must have a unique name."
            )

        elements[kde_name] = req_values

    return elements


def compare_element_names(
    file1,
    file2,
    output_path: str = "name_differences.txt",
) -> None:
    file1, file2 = Path(file1), Path(file2)

    elems1 = _extract_elements(_load_yaml(file1))
    elems2 = _extract_elements(_load_yaml(file2))

    names1, names2 = set(elems1), set(elems2)

    only_in_1 = sorted(names1 - names2)
    only_in_2 = sorted(names2 - names1)

    lines = []
    for name in only_in_1:
        lines.append(
            f"{name}"
        )
    for name in only_in_2:
        lines.append(
            f"{name}"
        )

    with open(output_path, "w", encoding="utf-8") as out:
        if lines:
            out.write("\n".join(lines) + "\n")
            print(
                f"[comparator] {len(lines)} name difference(s) -> '{output_path}'"
            )
        else:
            out.write("NO DIFFERENCES IN REGARDS TO ELEMENT NAMES\n")
            print(f"[comparator] No name differences -> '{output_path}'")


def compare_element_requirements(
    file1,
    file2,
    output_path: str = "requirement_differences.txt",
) -> None:
    file1, file2 = Path(file1), Path(file2)

    elems1 = _extract_elements(_load_yaml(file1))
    elems2 = _extract_elements(_load_yaml(file2))

    names1, names2 = set(elems1), set(elems2)

    rows = []

    # -- KDEs absent from one file entirely ----------------------------------
    for name in sorted(names1 - names2):
        rows.append(
            f"{name},"
            f"ABSENT-IN-{file2.name},"
            f"PRESENT-IN-{file1.name},"
            f"NA"
        )

    for name in sorted(names2 - names1):
        rows.append(
            f"{name},"
            f"ABSENT-IN-{file1.name},"
            f"PRESENT-IN-{file2.name},"
            f"NA"
        )

    # -- KDEs in both files - compare requirement values ---------------------
    for name in sorted(names1 & names2):
        reqs1, reqs2 = elems1[name], elems2[name]

        # Requirement value in file1 but missing from file2
        for req in sorted(reqs1 - reqs2):
            rows.append(
                f"{name},"
                f"ABSENT-IN-{file2.name},"
                f"PRESENT-IN-{file1.name},"
                f"{req}"
            )

        # Requirement value in file2 but missing from file1
        for req in sorted(reqs2 - reqs1):
            rows.append(
                f"{name},"
                f"ABSENT-IN-{file1.name},"
                f"PRESENT-IN-{file2.name},"
                f"{req}"
            )

    with open(output_path, "w", encoding="utf-8") as out:
        if rows:
            out.write("\n".join(rows) + "\n")
            print(
                f"[comparator] {len(rows)} requirement difference(s) -> "
                f"'{output_path}'"
            )
        else:
            out.write("NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n")
            print(
                f"[comparator] No requirement differences -> '{output_path}'"
            )


def main(yaml1, yaml2) -> None:
    compare_element_names(yaml1, yaml2, output_path="name_differences.txt")
    compare_element_requirements(
        yaml1, yaml2, output_path="requirement_differences.txt"
    )

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])