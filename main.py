import argparse
import os
import random
import sys
import time

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# =========================================================
# LOCAL PROJECT IMPORTS
# =========================================================
# Import at module level because Python does not allow `from models import *`
# inside a function.
CURRENT_DIR = os.path.abspath(os.getcwd())
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from loader import DataLoader
from models import (
    RecommenderModel,
    BaselineExplainer,
    VanillaKernelSHAP,
    LIMEExplainer,
    LeaveOneOutExplainer,
    SmartShapExplainer,
    SmartShapSentenceExplainer,
    SmartShapLOOEnsemble,
    BanzhafExplainer,
    RISEExplainer,
    RandomExplainer,
    AdaptiveSmartShapLOO,
    ScreenerSmartShap,
)
from xai_evaluator import XAIEvaluator


# =========================================================
# PATH SETUP
# =========================================================
def resolve_base_path(base_path_arg=None):
    r"""
    Resolve project root path.

    Priority:
    1. --base_path argument
    2. Current working directory

    Run from the project root folder, for example the local XAI project directory.
    """
    if base_path_arg:
        return os.path.abspath(base_path_arg)
    return os.path.abspath(os.getcwd())


def check_required_files(base_path):
    required_files = ["loader.py", "models.py", "xai_evaluator.py"]
    missing_files = [
        f for f in required_files
        if not os.path.exists(os.path.join(base_path, f))
    ]
    if missing_files:
        raise FileNotFoundError(
            "Thiếu các file bắt buộc trong project root: "
            + ", ".join(missing_files)
            + f"\nProject root hiện tại: {base_path}"
        )


# =========================================================
# SEED CONTROL
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# SMALL UTILITIES
# =========================================================
def safe_get(row, name, default=""):
    """Safely get field from pandas namedtuple row."""
    value = getattr(row, name, default)
    if value is None:
        return default
    return value


def get_query_for_row(row, fixed_query=None):
    """
    Query policy:
    - If --query is provided, use it for all samples.
    - Otherwise, use paper title as the query.
    """
    if fixed_query is not None and str(fixed_query).strip():
        return str(fixed_query).strip()

    title = safe_get(row, "title", "")
    if isinstance(title, str) and title.strip():
        return title.strip()

    return "scientific paper recommendation"


def get_document_for_row(row):
    """Use abstract as the recommended document."""
    abstract = safe_get(row, "abstract", "")
    if isinstance(abstract, str) and abstract.strip():
        return abstract.strip()

    text = safe_get(row, "text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return ""


def format_float(value, digits=4):
    try:
        if value is None or np.isnan(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def canonical_mode(mode):
    """
    Normalize old/internal mode names into paper-facing SmartSHAP names.
    Old aliases are kept so existing command lines still work.
    """
    aliases = {
        "adaptive_mh_loo": "adaptive_smartshap_loo",
        "adaptive_mh_loo_opt": "adaptive_smartshap_loo",
        "screener_mh": "screener_smartshap",
        "screener_mh_opt": "screener_smartshap",
        "loo_shap_opt": "loo_shap",
    }
    return aliases.get(mode, mode)


# =========================================================
# BUILD EXPLAINER
# =========================================================
def build_explainer(
    mode,
    model,
    sentences,
    query,
    document_text,
    ns_smart,
    max_segments,
    seed,
):
    mode = canonical_mode(mode)

    if mode == "smartshap":
        return SmartShapExplainer(
            model,
            sentences,
            query,
            max_samples=ns_smart,
            batch_size=32,
            random_state=seed,
            use_loo_calibration=True,
        )

    if mode == "smartshap_sentence":
        return SmartShapSentenceExplainer(
            model,
            document_text,
            query,
            max_segments=max_segments,
            max_samples=ns_smart,
            batch_size=32,
            random_state=seed,
            selection_mode="query_topk",
            use_loo_calibration=True,
        )

    if mode == "lime":
        return LIMEExplainer(model, sentences, query)

    if mode == "loo":
        return LeaveOneOutExplainer(model, sentences, query)

    if mode == "loo_shap":
        return SmartShapLOOEnsemble(model, sentences, query, alpha=0.7)

    if mode == "banzhaf":
        return BanzhafExplainer(model, sentences, query, random_state=seed)

    if mode == "rise":
        return RISEExplainer(model, sentences, query, random_state=seed)

    if mode == "vanilla_shap":
        return VanillaKernelSHAP(model, sentences, query)

    if mode == "random":
        return RandomExplainer(sentences, random_state=seed)

    if mode == "adaptive_smartshap_loo":
        return AdaptiveSmartShapLOO(model, sentences, query)

    if mode == "screener_smartshap":
        return ScreenerSmartShap(model, sentences, query, top_k=5)

    if mode == "baseline_kernel":
        return BaselineExplainer(model, sentences, query)

    raise ValueError(f"Unsupported mode: {mode}")


# =========================================================
# RUN ONE EXPLANATION
# =========================================================
def run_single_explanation(
    mode,
    model,
    sentences,
    query,
    document_text,
    ns_smart,
    ns_base,
    max_segments,
    seed,
):
    mode = canonical_mode(mode)

    explainer = build_explainer(
        mode=mode,
        model=model,
        sentences=sentences,
        query=query,
        document_text=document_text,
        ns_smart=ns_smart,
        max_segments=max_segments,
        seed=seed,
    )

    if mode in ["smartshap", "smartshap_sentence", "loo_shap", "adaptive_smartshap_loo", "screener_smartshap"]:
        values = explainer.explain(n_samples=ns_smart)

    elif mode == "banzhaf":
        values = explainer.explain(n_samples=ns_smart)

    elif mode == "rise":
        values = explainer.explain(n_samples=max(ns_smart, 500))

    elif mode in ["lime", "loo", "random"]:
        values = explainer.explain()

    elif mode in ["baseline_kernel", "vanilla_shap"]:
        values = explainer.explain(n_samples=ns_base)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return np.asarray(values, dtype=float).reshape(-1)


# =========================================================
# RUN ONE MODE
# =========================================================
def run_mode(args, mode, model, loader, evaluator, df_sampled, sample_size, writer):
    mode = canonical_mode(mode)
    print(f"\n[*] MODE: {mode} | samples={sample_size} | seed={args.seed}")

    results = []

    for idx, row in enumerate(df_sampled.itertuples(index=False), start=1):
        document_text = get_document_for_row(row)
        query = get_query_for_row(row, fixed_query=args.query)

        sentences = loader.preprocess_abstract(document_text)
        if len(sentences) < args.min_sentences:
            print(f"[SKIP] [{idx}/{sample_size}] abstract quá ngắn: {len(sentences)} câu")
            continue

        title = safe_get(row, "title", "Untitled")
        title = str(title) if title is not None else "Untitled"
        print(f"\n>>> [{idx}/{sample_size}] {title[:80]}")
        print(f"    Query: {query[:100]}")
        print(f"    Sentences: {len(sentences)}")

        try:
            # ---- RUN CURRENT EXPLAINER ----
            t0_wall = time.perf_counter()
            t0_cpu = time.process_time()

            current = run_single_explanation(
                mode=mode,
                model=model,
                sentences=sentences,
                query=query,
                document_text=document_text,
                ns_smart=args.ns_smart,
                ns_base=args.ns_base,
                max_segments=args.max_segments,
                seed=args.seed,
            )

            runtime_wall = time.perf_counter() - t0_wall
            runtime_cpu = time.process_time() - t0_cpu

            if current.size == 0:
                print("[SKIP] Explainer trả về attribution rỗng.")
                continue

            # If SmartSHAP-Sentence internally selected fewer sentences, align evaluation
            # to the same selected sentence list to avoid length mismatch.
            eval_sentences = sentences
            if mode == "smartshap_sentence" and current.size != len(sentences):
                tmp_explainer = build_explainer(
                    mode=mode,
                    model=model,
                    sentences=sentences,
                    query=query,
                    document_text=document_text,
                    ns_smart=args.ns_smart,
                    max_segments=args.max_segments,
                    seed=args.seed,
                )
                eval_sentences = tmp_explainer.sentences

            if current.size != len(eval_sentences):
                print(
                    f"[SKIP] Length mismatch: values={current.size}, "
                    f"sentences={len(eval_sentences)}"
                )
                continue

            # ---- FAITHFULNESS ----
            faith = evaluator.evaluate_faithfulness(
                query,
                eval_sentences,
                current,
                model,
                k=args.faith_k,
                use_abs=args.use_abs_faith,
            )

            # ---- BASELINE COMPARISON ----
            if mode != "baseline_kernel":
                gt = run_single_explanation(
                    mode="baseline_kernel",
                    model=model,
                    sentences=eval_sentences,
                    query=query,
                    document_text=" ".join(eval_sentences),
                    ns_smart=args.ns_smart,
                    ns_base=args.ns_base,
                    max_segments=args.max_segments,
                    seed=args.seed,
                )
                rank = evaluator.evaluate_rank_agreement(gt, current)
                err = evaluator.evaluate_error(gt, current)
            else:
                rank = np.nan
                err = {"mae": 0.0, "rmse": 0.0}

            # ---- STABILITY ----
            runs = []
            repeated_times = []

            for run_id in range(max(1, args.stability_runs)):
                seed_i = args.seed + run_id + 1
                t_run_wall = time.perf_counter()

                r = run_single_explanation(
                    mode=mode,
                    model=model,
                    sentences=eval_sentences,
                    query=query,
                    document_text=" ".join(eval_sentences),
                    ns_smart=args.ns_smart,
                    ns_base=args.ns_base,
                    max_segments=args.max_segments,
                    seed=seed_i,
                )

                repeated_times.append(time.perf_counter() - t_run_wall)
                runs.append(r)

            stability = evaluator.evaluate_stability(runs)

            res = {
                "comp": faith["comprehensiveness"],
                "suff": faith["sufficiency"],
                "rank": rank,
                "mae": err["mae"],
                "rmse": err["rmse"],
                "stab": stability,
                "time": runtime_wall,
                "cpu_time": runtime_cpu,
                "repeat_time": float(np.mean(repeated_times)) if repeated_times else np.nan,
            }

            results.append(res)

            step = len(results)
            writer.add_scalar(f"{mode}/Comprehensiveness", res["comp"], step)
            writer.add_scalar(f"{mode}/Sufficiency", res["suff"], step)
            writer.add_scalar(f"{mode}/RankAgreement", 0.0 if np.isnan(res["rank"]) else res["rank"], step)
            writer.add_scalar(f"{mode}/MAE", res["mae"], step)
            writer.add_scalar(f"{mode}/RMSE", res["rmse"], step)
            writer.add_scalar(f"{mode}/Stability", res["stab"], step)
            writer.add_scalar(f"{mode}/RuntimeWall", res["time"], step)
            writer.add_scalar(f"{mode}/RuntimeCPU", res["cpu_time"], step)

            print(
                f"Comp={format_float(res['comp'])} | "
                f"Suff={format_float(res['suff'])} | "
                f"Rank={format_float(res['rank'])} | "
                f"MAE={format_float(res['mae'])} | "
                f"RMSE={format_float(res['rmse'])} | "
                f"Stab={format_float(res['stab'])} | "
                f"Time={res['time']:.2f}s | CPU={res['cpu_time']:.2f}s"
            )

        except Exception as e:
            print(f"[!] Error at sample {idx}: {e}")
            continue

    print_summary(mode, results)
    return results


# =========================================================
# SUMMARY
# =========================================================
def print_summary(mode, results):
    if not results:
        print(f"\n[!] Không có kết quả hợp lệ cho mode: {mode}")
        return

    print("\n" + "=" * 24 + f" AVG RESULTS ({mode.upper()}) " + "=" * 24)
    print(f"Avg Comprehensiveness:            {np.mean([r['comp'] for r in results]):.4f}")
    print(f"Avg Sufficiency (drop):           {np.mean([r['suff'] for r in results]):.4f}")

    rank_values = [r["rank"] for r in results if not np.isnan(r["rank"])]
    if rank_values:
        print(f"Avg Rank Agreement vs Baseline:   {np.mean(rank_values):.4f}")
    else:
        print("Avg Rank Agreement vs Baseline:   N/A (reference mode)")

    print(f"Avg MAE vs Baseline:              {np.mean([r['mae'] for r in results]):.4f}")
    print(f"Avg RMSE vs Baseline:             {np.mean([r['rmse'] for r in results]):.4f}")
    print(f"Avg Stability:                    {np.mean([r['stab'] for r in results]):.4f}")
    print(f"Avg Runtime per paper:            {np.mean([r['time'] for r in results]):.2f}s")
    print(f"Avg CPU Time per paper:           {np.mean([r['cpu_time'] for r in results]):.2f}s")
    print(f"Avg Runtime repeated runs:        {np.mean([r['repeat_time'] for r in results]):.2f}s")
    print("=" * 78)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--mode", type=str, default="smartshap")
    parser.add_argument(
        "--all_modes",
        type=str,
        default="baseline_kernel,smartshap,smartshap_sentence,adaptive_smartshap_loo,loo,banzhaf,lime,rise,vanilla_shap",
        help="Comma-separated mode list used when --mode all.",
    )

    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--ns_smart", type=int, default=200)
    parser.add_argument("--ns_base", type=int, default=1000)
    parser.add_argument("--max_segments", type=int, default=12)
    parser.add_argument("--min_sentences", type=int, default=2)

    # If omitted, each paper title is used as query.
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--log_name", type=str, default=None)

    # XAI metrics
    parser.add_argument("--faith_k", type=int, default=1)
    parser.add_argument("--stability_runs", type=int, default=3)
    parser.add_argument("--use_abs_faith", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    base_path = resolve_base_path(args.base_path)
    check_required_files(base_path)

    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    exp_folder = args.log_name if args.log_name else f"run_{int(time.time())}"
    log_dir = os.path.join(base_path, "runs", exp_folder)
    os.makedirs(log_dir, exist_ok=True)

    if SummaryWriter is None:
        print("[WARN] Thiếu tensorboard. Bỏ qua ghi TensorBoard log.")
        writer = NullWriter()
    else:
        writer = SummaryWriter(log_dir)

    print(f"[*] Project root: {base_path}")
    print(f"[*] Log dir: {log_dir}")
    print(f"[*] Dataset: {args.dataset}")
    print(f"[*] Query policy: {'fixed query' if args.query else 'row title'}")

    loader = DataLoader(base_path=base_path)
    model = RecommenderModel()
    evaluator = XAIEvaluator()

    df = loader.load_dataset(args.dataset)
    if len(df) == 0:
        raise ValueError(f"Dataset rỗng: {args.dataset}")

    sample_size = min(args.samples, len(df))
    df_sampled = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    if args.mode == "all":
        modes = [canonical_mode(m.strip()) for m in args.all_modes.split(",") if m.strip()]
        # Remove duplicates while preserving order.
        modes = list(dict.fromkeys(modes))
    else:
        modes = [canonical_mode(args.mode)]

    all_results = {}
    for mode in modes:
        all_results[mode] = run_mode(
            args=args,
            mode=mode,
            model=model,
            loader=loader,
            evaluator=evaluator,
            df_sampled=df_sampled,
            sample_size=sample_size,
            writer=writer,
        )

    writer.close()


if __name__ == "__main__":
    main()
