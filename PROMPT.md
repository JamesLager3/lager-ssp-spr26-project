# Zero-Shot

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

# Few-Shot

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

TASK: You are a security engineer that needs to detect requirements for specific data elements. 
Identify Key Data Elements in security text, and output into a FLAT JSON SCHEMA.

CRITICAL RULES:
1. Each element MUST:
- be a noun (data entity)
- have associated requirement actions
2. The value of "requirements" MUST be a dictionary of "reqX" keys, not a list.
3. Ignore lines of code, version numbers, page numbers.

# Chain-Of-Thought

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