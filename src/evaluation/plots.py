"""Create one chart per metric from an evaluation summary JSON file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def create_plots(summary_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError("install matplotlib to generate evaluation plots") from exc

    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    variants = summary.get("variants", {})
    if not variants:
        raise ValueError("evaluation summary contains no variants")
    target = Path(out_dir) if out_dir else source.parent / "plots"
    target.mkdir(parents=True, exist_ok=True)

    metrics = {
        "mean_token_f1": "Mean answer token F1",
        "mean_exact_match": "Mean exact match",
        "mean_evidence_recall": "Mean evidence recall",
        "mean_evidence_precision": "Mean evidence precision",
        "mean_context_tokens": "Mean context tokens",
        "mean_latency_ms": "Mean end-to-end latency (ms)",
    }
    labels = list(variants)
    written = []
    stem = source.stem.removesuffix("_summary")
    for key, title in metrics.items():
        values = [variants[label].get(key) for label in labels]
        keep = [(label, value) for label, value in zip(labels, values) if value is not None]
        if not keep:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(keep) * 0.8), 5))
        ax.bar([label for label, _ in keep], [value for _, value in keep])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        path = target / f"{stem}_{key}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    paths = create_plots(args.summary, args.out_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
