"""Command-line prediction entry point for the retained hybrid detector."""
from __future__ import annotations

import argparse
import json

try:
    from .hybrid_classifier import HybridRumourDetectClass
except ImportError:
    from hybrid_classifier import HybridRumourDetectClass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rumor label and explanation for one text.")
    parser.add_argument("--model", default="models/ensemble.joblib", help="Path to retained local ensemble model")
    parser.add_argument("--retriever", default="models/retriever.joblib", help="Path to retriever artifact")
    parser.add_argument("--config", default="config.json", help="Path to API/config JSON")
    parser.add_argument("--text", required=True, help="Input tweet/text")
    parser.add_argument("--use-llm", action="store_true", help="Enable DeepSeek review for low-confidence samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clf = HybridRumourDetectClass(
        model_path=args.model,
        retriever_path=args.retriever,
        config_path=args.config,
        use_llm=args.use_llm,
    )
    result = clf.predict_with_reason(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
