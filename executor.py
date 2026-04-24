import subprocess
import pandas as pd
import json
import re
from pathlib import Path
import zipfile
import tempfile
import shutil
import os

CONTROL_MAPPING = {
    # ── Image / registry ──────────────────────────────────────────────────
    "C-0001": ["registry", "forbidden", "image", "container registry"],
    "C-0075": ["imagepullpolicy", "pullpolicy", "latesttag", "latest"],
    "C-0078": ["allowedregistry", "allowed registry"],
    "C-0221": ["ecr", "image scanning", "vulnerability scanning"],
    "C-0236": ["signature", "verify image"],
    "C-0237": ["signature"],
    "C-0243": ["azure defender", "acr scanning"],
    "C-0253": ["deprecated", "k8s.gcr.io", "deprecated registry"],

    # ── Resource limits / requests ────────────────────────────────────────
    "C-0004": ["memorylimit", "memoryrequest", "resources.limits.memory", "resources.requests.memory"],
    "C-0009": ["resources.limits", "resourcelimits", "limitrange"],
    "C-0050": ["cpulimit", "cpurequest", "resources.limits.cpu", "resources.requests.cpu"],
    "C-0268": ["cpu request", "resources.requests.cpu"],
    "C-0269": ["memory request", "resources.requests.memory"],
    "C-0270": ["cpu limit", "resources.limits.cpu"],
    "C-0271": ["memory limit", "resources.limits.memory"],

    # ── Container execution / commands ────────────────────────────────────
    "C-0002": ["exec", "command execution", "kubectl exec"],
    "C-0062": ["sudo", "entrypoint"],

    # ── Privilege escalation / security context ───────────────────────────
    "C-0013": ["runasnonroot", "runasuser", "runasgroup", "non-root", "nonroot"],
    "C-0016": ["allowprivilegeescalation", "privilege escalation"],
    "C-0017": ["readonlyrootfilesystem", "immutable", "readonly filesystem"],
    "C-0046": ["capabilities", "capadd", "capabilities.add"],
    "C-0055": ["seccomp", "apparmor", "linux hardening", "selinux"],
    "C-0057": ["privileged", "securitycontext.privileged"],
    "C-0193": ["privileged container"],
    "C-0197": ["allowprivilegeescalation"],
    "C-0198": ["runasroot", "root container"],
    "C-0199": ["net_raw", "netraw"],
    "C-0200": ["capabilities.add"],
    "C-0201": ["capabilities"],
    "C-0210": ["seccompprofile", "seccomp"],
    "C-0211": ["securitycontext", "podsecuritycontext"],
    "C-0213": ["privileged"],
    "C-0217": ["allowprivilegeescalation"],
    "C-0218": ["runasroot"],

    # ── Host namespaces / network ─────────────────────────────────────────
    "C-0038": ["hostpid", "hostipc", "pid", "ipc"],
    "C-0041": ["hostnetwork"],
    "C-0044": ["hostport", "containerport.hostport"],
    "C-0194": ["hostpid"],
    "C-0195": ["hostipc"],
    "C-0196": ["hostnetwork"],
    "C-0214": ["hostpid"],
    "C-0215": ["hostipc"],
    "C-0216": ["hostnetwork"],
    "C-0275": ["hostpid"],
    "C-0276": ["hostipc"],

    # ── HostPath / volumes ────────────────────────────────────────────────
    "C-0045": ["hostpath", "writable", "volumemounts"],
    "C-0048": ["hostpath", "volumes.hostpath"],
    "C-0074": ["socket", "docker.sock", "containerd.sock", "runtime socket"],
    "C-0203": ["hostpath"],
    "C-0257": ["pvc", "persistentvolumeclaim"],
    "C-0264": ["persistentvolume", "pv encryption"],

    # ── Secrets / credentials ─────────────────────────────────────────────
    "C-0012": ["credential", "password", "token", "apikey", "configmap secret"],
    "C-0015": ["list secrets", "secrets access"],
    "C-0066": ["etcd encryption", "secret encryption"],
    "C-0141": ["encryption-provider-config"],
    "C-0142": ["encryptionconfig", "encryption provider"],
    "C-0186": ["secrets", "secret access"],
    "C-0207": ["secret env", "secretkeyref", "envfrom secret"],
    "C-0208": ["external secret", "vault", "secret store"],
    "C-0234": ["external secret", "aws secret"],
    "C-0244": ["kubernetes secrets encrypted"],
    "C-0255": ["secret access", "secrets"],
    "C-0259": ["credential access"],

    # ── Service accounts ──────────────────────────────────────────────────
    "C-0020": ["serviceaccountname", "mount service principal"],
    "C-0034": ["automountserviceaccounttoken", "automount"],
    "C-0053": ["serviceaccount", "service account token"],
    "C-0125": ["serviceaccount admission"],
    "C-0135": ["service-account-lookup"],
    "C-0136": ["service-account-key-file"],
    "C-0189": ["default serviceaccount", "default service account"],
    "C-0190": ["serviceaccounttoken", "automountserviceaccounttoken"],
    "C-0225": ["eks serviceaccount", "irsa"],
    "C-0239": ["aks serviceaccount", "workload identity"],
    "C-0261": ["serviceaccount token mounted"],
    "C-0282": ["token creation", "serviceaccount token create"],

    # ── RBAC ──────────────────────────────────────────────────────────────
    "C-0007": ["delete", "role", "clusterrole", "rbac"],
    "C-0035": ["clusteradmin", "cluster-admin", "adminrole"],
    "C-0063": ["portforward", "portforwarding"],
    "C-0065": ["impersonation", "impersonate"],
    "C-0088": ["rbac", "rolebinding", "clusterrolebinding"],
    "C-0118": ["authorization-mode", "alwaysallow"],
    "C-0119": ["authorization-mode node"],
    "C-0120": ["authorization-mode rbac"],
    "C-0185": ["cluster-admin", "clusteradmin"],
    "C-0187": ["wildcard", "verbs: ['*']", "resources: ['*']"],
    "C-0188": ["create pods", "pods/create"],
    "C-0191": ["impersonate", "escalate", "bind"],
    "C-0232": ["iam authenticator", "aws-iam"],
    "C-0241": ["azure rbac", "aad"],
    "C-0246": ["system:masters"],
    "C-0262": ["anonymous", "rolebinding", "system:anonymous"],
    "C-0265": ["system:authenticated", "elevated role"],
    "C-0267": ["cluster takeover", "admin clusterrole"],
    "C-0272": ["administrative role", "cluster-admin binding"],
    "C-0278": ["persistentvolume create", "pv create"],
    "C-0279": ["nodes/proxy"],
    "C-0280": ["certificatesigningrequests", "csr approval"],
    "C-0281": ["webhook", "mutatingwebhookconfiguration", "validatingwebhookconfiguration"],

    # ── Network policies ──────────────────────────────────────────────────
    "C-0030": ["ingress", "egress", "networkpolicy"],
    "C-0049": ["networkpolicy", "network mapping"],
    "C-0054": ["cluster networking", "internal network"],
    "C-0205": ["cni", "network policy support"],
    "C-0206": ["namespace networkpolicy", "all namespaces"],
    "C-0230": ["networkpolicy", "gke network"],
    "C-0240": ["networkpolicy", "aks network"],
    "C-0260": ["missing networkpolicy", "no networkpolicy"],

    # ── Ingress / exposure ────────────────────────────────────────────────
    "C-0021": ["exposed", "dashboard", "sensitive interface", "nodeport"],
    "C-0256": ["externalip", "loadbalancer", "nodeport", "external facing"],
    "C-0263": ["ingress tls", "tls termination"],
    "C-0266": ["gateway", "istio ingress", "gateway api"],

    # ── API server configuration ──────────────────────────────────────────
    "C-0005": ["insecure-port", "insecureport", "--insecure-port"],
    "C-0113": ["anonymous-auth", "--anonymous-auth"],
    "C-0114": ["token-auth-file", "--token-auth-file"],
    "C-0115": ["denyserviceexternalips"],
    "C-0116": ["kubelet-client-certificate", "kubelet-client-key"],
    "C-0117": ["kubelet-certificate-authority"],
    "C-0121": ["eventratelimit"],
    "C-0122": ["alwaysadmit"],
    "C-0123": ["alwayspullimages"],
    "C-0124": ["securitycontextdeny", "podsecuritypolicy"],
    "C-0126": ["namespacelifecycle"],
    "C-0127": ["noderestriction"],
    "C-0128": ["secure-port"],
    "C-0129": ["profiling", "--profiling"],
    "C-0134": ["request-timeout"],
    "C-0137": ["etcd-certfile", "etcd-keyfile"],
    "C-0138": ["tls-cert-file", "tls-private-key-file"],
    "C-0139": ["client-ca-file"],
    "C-0140": ["etcd-cafile"],
    "C-0143": ["cipher", "cryptographic cipher"],
    "C-0277": ["cipher", "gke cipher"],
    "C-0283": ["denyserviceexternalips", "--denyserviceexternalips"],

    # ── Audit logging ─────────────────────────────────────────────────────
    "C-0067": ["audit", "auditlog", "audit-log"],
    "C-0130": ["audit-log-path"],
    "C-0131": ["audit-log-maxage"],
    "C-0132": ["audit-log-maxbackup"],
    "C-0133": ["audit-log-maxsize"],
    "C-0160": ["audit policy"],
    "C-0161": ["audit policy"],
    "C-0254": ["audit log", "aks audit"],

    # ── Kubelet configuration ─────────────────────────────────────────────
    "C-0069": ["anonymous-auth kubelet", "kubelet anonymous"],
    "C-0070": ["kubelet tls", "kubelet-client-tls"],
    "C-0172": ["kubelet anonymous-auth"],
    "C-0173": ["kubelet authorization-mode"],
    "C-0174": ["kubelet client-ca-file"],
    "C-0175": ["read-only-port", "readonlyport"],
    "C-0176": ["streaming-connection-idle-timeout"],
    "C-0177": ["protect-kernel-defaults"],
    "C-0178": ["iptables", "make-iptables-util-chains"],
    "C-0179": ["hostname-override"],
    "C-0180": ["event-qps"],
    "C-0181": ["kubelet tls-cert-file"],
    "C-0182": ["rotate-certificates"],
    "C-0183": ["rotatekubeletservercertificate"],
    "C-0184": ["kubelet cipher"],
    "C-0284": ["pid limit", "podpidsLimit"],

    # ── File permissions (CIS benchmarks) ────────────────────────────────
    "C-0092": ["apiserver spec permissions", "kube-apiserver.yaml permissions"],
    "C-0093": ["apiserver spec ownership", "kube-apiserver.yaml ownership"],
    "C-0094": ["controller-manager spec permissions", "kube-controller-manager.yaml permissions"],
    "C-0095": ["controller-manager spec ownership"],
    "C-0096": ["scheduler spec permissions", "kube-scheduler.yaml permissions"],
    "C-0097": ["scheduler spec ownership"],
    "C-0098": ["etcd spec permissions", "etcd.yaml permissions"],
    "C-0099": ["etcd spec ownership"],
    "C-0100": ["cni file permissions"],
    "C-0101": ["cni file ownership"],
    "C-0102": ["etcd data permissions", "etcd data directory"],
    "C-0103": ["etcd data ownership"],
    "C-0104": ["admin.conf permissions"],
    "C-0105": ["admin.conf ownership"],
    "C-0106": ["scheduler.conf permissions"],
    "C-0107": ["scheduler.conf ownership"],
    "C-0108": ["controller-manager.conf permissions"],
    "C-0109": ["controller-manager.conf ownership"],
    "C-0110": ["pki ownership", "kubernetes pki"],
    "C-0111": ["pki cert permissions"],
    "C-0112": ["pki key permissions"],
    "C-0162": ["kubelet service permissions"],
    "C-0163": ["kubelet service ownership"],
    "C-0164": ["proxy kubeconfig permissions"],
    "C-0165": ["proxy kubeconfig ownership"],
    "C-0166": ["kubelet.conf permissions"],
    "C-0167": ["kubelet.conf ownership"],
    "C-0168": ["ca file permissions", "certificate authorities permissions"],
    "C-0169": ["ca file ownership"],
    "C-0170": ["kubelet config.yaml permissions"],
    "C-0171": ["kubelet config.yaml ownership"],
    "C-0235": ["kubelet configuration permissions"],
    "C-0238": ["kubeconfig permissions"],

    # ── Admission controllers ─────────────────────────────────────────────
    "C-0036": ["validatingwebhook", "admission validating"],
    "C-0039": ["mutatingwebhook", "admission mutating"],
    "C-0068": ["podsecuritypolicy", "psp"],
    "C-0192": ["policy control", "opa", "kyverno", "gatekeeper"],

    # ── Namespaces / workload hygiene ─────────────────────────────────────
    "C-0061": ["default namespace", "namespace: default"],
    "C-0073": ["naked pod", "no deployment", "bare pod"],
    "C-0076": ["label", "labels"],
    "C-0077": ["app.kubernetes.io", "common label"],
    "C-0209": ["namespace", "boundary"],
    "C-0212": ["default namespace"],

    # ── Probes ────────────────────────────────────────────────────────────
    "C-0018": ["readinessprobe", "readiness probe"],
    "C-0056": ["livenessprobe", "liveness probe"],

    # ── SSH ───────────────────────────────────────────────────────────────
    "C-0042": ["ssh", "port 22"],

    # ── CronJob ───────────────────────────────────────────────────────────
    "C-0026": ["cronjob", "cron"],

    # ── DNS ───────────────────────────────────────────────────────────────
    "C-0037": ["coredns", "dns poisoning"],

    # ── Metadata API ─────────────────────────────────────────────────────
    "C-0052": ["metadata api", "imds", "169.254.169.254"],

    # ── TLS / certificates ────────────────────────────────────────────────
    "C-0153": ["etcd cert-file", "etcd key-file"],
    "C-0154": ["client-cert-auth"],
    "C-0155": ["auto-tls"],
    "C-0156": ["peer-cert-file", "peer-key-file"],
    "C-0157": ["peer-client-cert-auth"],
    "C-0158": ["peer-auto-tls"],
    "C-0159": ["etcd ca", "unique ca"],
    "C-0231": ["https loadbalancer", "tls certificate"],
    "C-0245": ["https loadbalancer", "aks tls"],

    # ── Vulnerabilities ───────────────────────────────────────────────────
    "C-0083": ["critical vulnerability", "cve external"],
    "C-0084": ["rce", "remote code execution"],
    "C-0085": ["excessive vulnerabilities"],

    # ── Cloud (AWS/EKS) ───────────────────────────────────────────────────
    "C-0222": ["ecr user access"],
    "C-0223": ["ecr cluster access"],
    "C-0227": ["control plane endpoint", "eks endpoint"],
    "C-0228": ["private endpoint", "eks private"],
    "C-0229": ["private nodes", "eks nodes"],
    "C-0233": ["fargate"],

    # ── Cloud (Azure/AKS) ─────────────────────────────────────────────────
    "C-0242": ["multi-tenant", "hostile tenant"],
    "C-0247": ["aks control plane endpoint"],
    "C-0248": ["aks private nodes"],
    "C-0250": ["acr cluster access"],
    "C-0251": ["acr user access"],
    "C-0252": ["aks private endpoint"],

    # ── ConfigMap access ──────────────────────────────────────────────────
    "C-0258": ["configmap", "configmap access"],

    # ── Misc ──────────────────────────────────────────────────────────────
    "C-0014": ["dashboard", "kubernetes dashboard"],
    "C-0031": ["delete events", "events delete"],
    "C-0058": ["symlink", "cve-2021-25741"],
    "C-0059": ["nginx snippet", "cve-2021-25742"],
    "C-0079": ["cve-2022-0185", "kernel escape"],
    "C-0081": ["argocd", "cve-2022-24348"],
    "C-0087": ["containerd escape", "cve-2022-23648"],
    "C-0089": ["aggregated api", "cve-2022-3172"],
    "C-0090": ["grafana", "cve-2022-39328"],
    "C-0091": ["kyverno", "cve-2022-47633"],
    "C-0202": ["windows hostprocess", "hostprocess"],
    "C-0226": ["container os", "cos", "bottlerocket"],
    "C-0273": ["kubernetes version", "outdated version"],
    "C-0274": ["authenticated service", "service auth"],
}


def read_input_files(name_diff_file: str, req_diff_file: str) -> list[str]:
    differences = []

    # --- Name differences file ---
    with open(name_diff_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not (len(lines) == 1 and "NO DIFFERENCES" in lines[0].upper()):
        differences.extend(lines)

    # --- Requirement differences file ---
    with open(req_diff_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or "NO DIFFERENCES" in line.upper():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            name = parts[0]
            req  = parts[3]

            if req.upper() == "NA":
                differences.append(name)
            else:
                differences.append(f"{name}:{req}")

    return differences


def generate_control_file(name_diff_file: str,
                           req_diff_file: str,
                           output_file: str = "controls.txt") -> str:
    differences = read_input_files(name_diff_file, req_diff_file)

    if not differences:
        with open(output_file, "w") as f:
            f.write("NO DIFFERENCES FOUND")
        print("[INFO] No differences detected — controls.txt: NO DIFFERENCES FOUND")
        return output_file

    haystack = " ".join(differences).lower()
    matched = set()

    for control_id, keywords in CONTROL_MAPPING.items():
        if any(kw in haystack for kw in keywords):
            matched.add(control_id)

    controls = sorted(matched)

    if controls:
        with open(output_file, "w") as f:
            f.write("\n".join(controls))
    else:
        with open(output_file, "w") as f:
            f.write("NO DIFFERENCES FOUND")

    return output_file


def run_kubescape(control_file: str,
                  yaml_archive: str = "project-yamls.zip") -> pd.DataFrame:
    if not Path(yaml_archive).exists():
        raise FileNotFoundError(f"YAML archive not found: {yaml_archive}")

    kubescape = shutil.which("kubescape")
    with open(control_file, "r") as f:
        file_content = f.read().strip()

    tmp_dir = tempfile.mkdtemp(prefix="kubescape_yamls_")
    tmp_prefix = os.path.join(tmp_dir, "")
    
    try:
        with zipfile.ZipFile(yaml_archive, "r") as zf:
            zf.extractall(tmp_dir)

        if file_content == "NO DIFFERENCES FOUND":
            cmd = [kubescape, "scan", tmp_dir, "--format", "json"]
            print("[INFO] Running Kubescape with ALL controls.")
        else:
            controls = [ln.strip() for ln in file_content.splitlines() if ln.strip()]
            cmd = [kubescape, "scan", "control", ",".join(controls), tmp_dir, "--format", "json"]
            print(f"[INFO] Running Kubescape with controls: {controls}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if not result.stdout:
            raise RuntimeError(f"Kubescape scan failed:\n{result.stderr}")

        data = json.loads(result.stdout)

        control_metadata = data.get("summaryDetails", {}).get("controls", {})

        rows = []
        seen_entries = set() 

        for resource_result in data.get("results", []):
            res_id = resource_result.get("resourceID", "")
            
            # Clean the FilePath
            clean_path = res_id
            if clean_path.startswith(tmp_prefix):
                clean_path = clean_path.replace(tmp_prefix, "", 1)
            
            match = re.search(r"^(.*?\.(?:yaml|yml))", clean_path, re.IGNORECASE)
            if match:
                clean_path = match.group(1)

            # Iterate through the controls failed by THIS specific resource
            for ctrl in resource_result.get("controls", []):
                if ctrl.get("status", {}).get("status") == "failed":
                    cid = ctrl.get("controlID", "")
                    
                    # Deduplication check: (Path + Control ID)
                    entry_key = (clean_path, cid)
                    if entry_key in seen_entries:
                        continue
                    
                    meta = control_metadata.get(cid, {})
                    counters = meta.get("ResourceCounters", {})

                    rows.append({
                        "FilePath":         clean_path,
                        "Severity":         meta.get("severity", ""),
                        "Control name":     meta.get("name", ""),
                        "Failed resources": counters.get("failedResources", 0),
                        "All Resources":    (counters.get("passedResources", 0) + 
                                             counters.get("failedResources", 0) + 
                                             counters.get("skippedResources", 0)),
                        "Compliance score": meta.get("complianceScore", ""),
                    })
                    
                    seen_entries.add(entry_key)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    required_columns = [
        "FilePath", "Severity", "Control name",
        "Failed resources", "All Resources", "Compliance score"
    ]
    
    if df.empty:
        return pd.DataFrame(columns=required_columns)

    return df[required_columns]


def generate_csv(df: pd.DataFrame,
                 output_csv: str = "kubescape_results.csv") -> str:
    """
    Writes the scan DataFrame to a CSV file with the required headers.
    Returns the path to the CSV file.
    """
    required_columns = [
        "FilePath", "Severity", "Control name",
        "Failed resources", "All Resources", "Compliance score"
    ]
    df = df.reindex(columns=required_columns)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] CSV written → {output_csv}")
    return output_csv


def main(name_diff_file: str, req_diff_file: str):
    if os.path.exists('controls.txt'):
        os.remove('controls.txt')
    control_file = generate_control_file(name_diff_file, req_diff_file)
    df           = run_kubescape(control_file)
    csv_path     = generate_csv(df)
    return df, csv_path