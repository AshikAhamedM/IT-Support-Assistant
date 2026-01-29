from datasets import Dataset
from ragas import evaluate

dataset = Dataset.from_dict({
    "question": ["VPN is not connecting"],
    "answer": ["Check internet, credentials, and VPN client version"],
    "contexts": [["VPN requires active internet and valid credentials"]],
    "ground_truth": ["Check VPN credentials and internet connection"]
})

result = evaluate(dataset)
print(result)
