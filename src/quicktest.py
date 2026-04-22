import json
from collections import Counter

with open("outputs/results/hybrid_gpt-4o-mini_n3_08da3a0b.json") as f:
    data = json.load(f)

for aid, pred in data["predictions"].items():
    verdicts = Counter(s["verdict"] for s in pred["all_spans"])
    n_critiques = len(pred["stage_outputs"].get("stage2_critique_parsed", []))
    print(f"{aid}: verdicts={dict(verdicts)}, critiques_parsed={n_critiques}")