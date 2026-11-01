"""Write a seeded sample of MedMCQA train as dvx JSONL (prompt/response)."""

import argparse
import json
import random
from pathlib import Path

from dvx.evals.registry import get_suite

_LETTERS = "ABCD"


def medmcqa_row_to_example(row: dict) -> dict:
    options = [row["opa"], row["opb"], row["opc"], row["opd"]]
    lines = [f"{letter}. {opt}" for letter, opt in zip(_LETTERS, options)]
    prompt = (
        f"{row['question']}\n\n" + "\n".join(lines) + "\n\nAnswer with the letter of the correct option."
    )
    cop = int(row["cop"])
    response = f"{_LETTERS[cop]}. {options[cop]}"
    exp = row.get("exp")
    if isinstance(exp, str) and exp.strip():
        response += f"\n\nExplanation: {exp}"
    return {"prompt": prompt, "response": response}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    bench = next(b for b in get_suite("medical-qa").benchmarks if b.name == "medmcqa")
    ds = load_dataset(bench.hf_id, split="train", revision=bench.revision)
    indices = sorted(random.Random(args.seed).sample(range(len(ds)), args.n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i in indices:
            f.write(json.dumps(medmcqa_row_to_example(ds[i])) + "\n")
    print(f"wrote {len(indices)} examples to {args.out}")


if __name__ == "__main__":
    main()
