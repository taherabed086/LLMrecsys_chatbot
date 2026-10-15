from recsys_chat.agent import run_agent
from recsys_chat.config import DATASETS
from recsys_chat.data import load_items

dataset = "beauty"
items = load_items(dataset)
label = DATASETS[dataset]["label"] if isinstance(DATASETS[dataset], dict) else DATASETS[dataset]

tests = [
    "something for dry skin",
    "بدي شي للبشرة الجافة",
    "do you sell car tires?",
]
for text in tests:
    reply, results, level = run_agent([{"role": "user", "content": text}], items, dataset, label)
    print("\nQ:", text)
    print("A:", reply)
    print("confidence:", level, "| titles:", None if results is None else list(results["title"][:3]))