"""Generate proposal-defined Role 5 ablation configurations."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def ablation_configs(base: dict) -> dict[str, dict]:
    """Return independent configs for signal, search-strategy and budget ablations."""
    specs = {}
    for weight in ("alpha", "beta", "eta", "gamma"):
        cfg = copy.deepcopy(base)
        cfg[weight] = 0.0
        specs[f"without_{weight}"] = cfg
    for name, overrides in {
        "uniform_cost": {"W": 0.0, "use_g": True},
        "greedy_best_first": {"W": 1.0, "use_g": False},
        "astar_style": {"W": 1.0, "use_g": True},
        "weighted_astar": {"W": 1.5, "use_g": True},
    }.items():
        cfg = copy.deepcopy(base)
        cfg.update(overrides)
        specs[name] = cfg
    cfg = copy.deepcopy(base)
    cfg["tau"] = -1.0
    specs["no_path_relevance_pruning"] = cfg
    for beam_k in (1, 3, 5, 10):
        cfg = copy.deepcopy(base)
        cfg["beam_k"] = beam_k
        specs[f"beam_{beam_k}"] = cfg
    for margin in (0.1, 0.25, 0.5):
        cfg = copy.deepcopy(base)
        cfg["beam_k"] = None
        cfg["beam_margin"] = margin
        specs[f"beam_margin_{str(margin).replace('.', '_')}"] = cfg
    cfg = copy.deepcopy(base)
    cfg["epsilon"] = -1e9
    specs["fixed_budget_no_utility_stop"] = cfg
    for budget in (250, 500, 1000, 2000, 4000):
        cfg = copy.deepcopy(base)
        cfg["budget_B"] = budget
        specs[f"budget_{budget}"] = cfg
    # Context ablation: omit source paragraphs and serialize only graph triples.
    specs["context_without_source_sentences"] = {
        "role": "context", "max_sources_per_edge": 0, "tokenizer": "regex_wordpunct_v1"
    }
    return specs


def write_ablation_configs(base_path: str = "configs/traversal.yaml", out_dir: str = "configs/ablations") -> list[Path]:
    base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8")) or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, config in ablation_configs(base).items():
        # The source-sentence ablation belongs to Role 6's context config, not traversal.
        target_dir = Path("configs/context_ablations") if config.get("role") == "context" else out
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(config)
        payload.pop("role", None)
        path = target_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/traversal.yaml")
    parser.add_argument("--out-dir", default="configs/ablations")
    args = parser.parse_args(argv)
    paths = write_ablation_configs(args.base, args.out_dir)
    print(f"wrote {len(paths)} ablation configs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
