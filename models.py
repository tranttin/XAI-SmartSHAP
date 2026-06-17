import itertools
import math
import os
import re

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler

try:
    from huggingface_hub import login
except ImportError:
    login = None


# =========================================================
# Optional Hugging Face login from environment variable
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN and login is not None:
    try:
        login(HF_TOKEN)
    except Exception as e:
        print(f"[WARN] Hugging Face login failed: {e}")


# =========================================================
# 1. RECOMMENDATION MODEL
# =========================================================
class RecommenderModel:
    def __init__(
        self,
        bi_encoder_name="BAAI/bge-small-en",
        cross_encoder_name="BAAI/bge-reranker-base",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[*] Khởi tạo Recommender "
            f"(Bi-encoder: {bi_encoder_name}, Cross-encoder: {cross_encoder_name}) "
            f"trên {self.device}"
        )

        self.bi_encoder = SentenceTransformer(bi_encoder_name).to(self.device)
        self.cross_encoder = CrossEncoder(cross_encoder_name).to(self.device)

    def get_embedding(self, text):
        """Encode one text into a tensor embedding."""
        text = "" if text is None else str(text)
        return self.bi_encoder.encode(text, convert_to_tensor=True)

    def encode_texts(self, texts, batch_size=32, convert_to_numpy=True):
        """Encode a batch of texts."""
        if texts is None or len(texts) == 0:
            return np.array([], dtype=np.float32) if convert_to_numpy else []

        safe_texts = ["" if t is None else str(t) for t in texts]
        return self.bi_encoder.encode(
            safe_texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=False,
        )

    def predict_score(self, query_text, sentences):
        """
        Bi-encoder cosine similarity between query and document.

        `sentences` may be:
        - list[str]
        - str
        """
        if sentences is None:
            return 0.0

        if isinstance(sentences, list):
            doc_text = " ".join(str(s) for s in sentences if str(s).strip()).strip()
        else:
            doc_text = str(sentences).strip()

        query_text = "" if query_text is None else str(query_text).strip()

        if not query_text or not doc_text:
            return 0.0

        q_emb = self.get_embedding(query_text).cpu().detach().numpy().reshape(-1)
        d_emb = self.get_embedding(doc_text).cpu().detach().numpy().reshape(-1)

        return _safe_cosine_from_embeddings(q_emb, d_emb)

    def predict_score_batch(self, query_text, texts, batch_size=32):
        """
        Vectorized cosine similarity between one query and many documents.
        Empty documents are assigned score 0.0 instead of embedding an empty string.
        """
        if texts is None or len(texts) == 0:
            return np.array([], dtype=float)

        query_text = "" if query_text is None else str(query_text).strip()
        if not query_text:
            return np.zeros(len(texts), dtype=float)

        safe_texts = ["" if t is None else str(t).strip() for t in texts]
        valid_idx = [i for i, t in enumerate(safe_texts) if t]

        scores = np.zeros(len(safe_texts), dtype=float)
        if not valid_idx:
            return scores

        valid_texts = [safe_texts[i] for i in valid_idx]

        query_emb = self.encode_texts([query_text], batch_size=1, convert_to_numpy=True)
        query_emb = np.asarray(query_emb, dtype=np.float32)
        if query_emb.ndim == 2:
            query_emb = query_emb[0]
        query_emb = query_emb.reshape(-1)

        doc_embs = self.encode_texts(valid_texts, batch_size=batch_size, convert_to_numpy=True)
        doc_embs = np.asarray(doc_embs, dtype=np.float32)
        if doc_embs.ndim == 1:
            doc_embs = doc_embs.reshape(1, -1)

        q_norm = np.linalg.norm(query_emb)
        if q_norm < 1e-12:
            return scores

        d_norms = np.linalg.norm(doc_embs, axis=1)
        valid_doc = d_norms > 1e-12

        local_scores = np.zeros(len(valid_texts), dtype=float)
        if np.any(valid_doc):
            local_scores[valid_doc] = (
                doc_embs[valid_doc] @ query_emb
            ) / (d_norms[valid_doc] * q_norm)

        for pos, idx in enumerate(valid_idx):
            scores[idx] = float(local_scores[pos])

        return scores.astype(float)

    def predict_score_cross_batch(self, query_text, texts, batch_size=16):
        """
        Cross-encoder relevance score between query and many documents.
        """
        if texts is None or len(texts) == 0:
            return np.array([], dtype=float)

        query_text = "" if query_text is None else str(query_text).strip()
        safe_texts = ["" if t is None else str(t).strip() for t in texts]

        if not query_text:
            return np.zeros(len(safe_texts), dtype=float)

        pairs = [[query_text, doc] for doc in safe_texts]
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=float)

    def rerank(self, query_text, candidate_documents, batch_size=16):
        """Cross-encoder re-ranking for candidate documents."""
        if candidate_documents is None or len(candidate_documents) == 0:
            return np.array([], dtype=float)
        return self.predict_score_cross_batch(
            query_text,
            candidate_documents,
            batch_size=batch_size,
        )


# =========================================================
# 2. SHARED HELPERS
# =========================================================
def _safe_cosine_from_embeddings(query_emb, doc_emb):
    q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
    d = np.asarray(doc_emb, dtype=np.float32).reshape(-1)

    q_norm = np.linalg.norm(q)
    d_norm = np.linalg.norm(d)

    if q_norm < 1e-12 or d_norm < 1e-12:
        return 0.0

    return float(np.dot(q, d) / (q_norm * d_norm))


def _kernel_shap_weight(m, s):
    """KernelSHAP coalition weight for m features and subset size s."""
    if s <= 0 or s >= m:
        return 1e-8

    try:
        denom = math.comb(m, s) * s * (m - s)
        return float((m - 1) / denom) if denom != 0 else 1e-8
    except OverflowError:
        return 1e-12


def _max_abs_normalize(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values

    max_abs = float(np.max(np.abs(values)))
    if max_abs < 1e-12:
        return np.zeros_like(values, dtype=float)

    return values / max_abs


def _l1_normalize(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values

    denom = float(np.sum(np.abs(values)))
    if denom < 1e-12:
        return np.zeros_like(values, dtype=float)

    return values / denom


def generate_shap_samples(no_sentences, max_samples=2000, random_state=42):
    """
    Structured coalition sampler for sentence-level SHAP-style regression.

    Sampling design:
    - always include singleton coalitions and full coalition;
    - use exhaustive coalitions for very short documents;
    - prioritize pairwise and triple-wise coalitions;
    - fill the remaining budget with random higher-order coalitions.
    """
    rng = np.random.default_rng(random_state)

    if no_sentences <= 0:
        return []

    max_samples = int(max_samples) if max_samples is not None else 2000
    if max_samples <= 0:
        return []

    combos = set()
    all_idx = tuple(range(no_sentences))

    for i in range(no_sentences):
        combos.add((i,))
    combos.add(all_idx)

    if no_sentences <= 8:
        for r in range(1, no_sentences + 1):
            for c in itertools.combinations(range(no_sentences), r):
                combos.add(tuple(c))
        return sorted(combos, key=lambda x: (len(x), x))[:max_samples]

    for r in [2, 3]:
        for c in itertools.combinations(range(no_sentences), r):
            combos.add(tuple(c))
            if len(combos) >= max_samples:
                return sorted(combos, key=lambda x: (len(x), x))[:max_samples]

    attempts = 0
    max_attempts = max_samples * 10

    while len(combos) < max_samples and attempts < max_attempts:
        attempts += 1

        if no_sentences <= 4:
            size = no_sentences
        else:
            size = int(rng.integers(4, no_sentences))

        sample = tuple(sorted(rng.choice(no_sentences, size=size, replace=False)))
        combos.add(sample)

    return sorted(combos, key=lambda x: (len(x), x))[:max_samples]


def fit_shap_model_from_rows(rows, no_sentences):
    """Fit weighted linear surrogate and return raw/max-abs-normalized coefficients."""
    if not rows or no_sentences <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), None

    num_rows = len(rows)
    X = np.zeros((num_rows, no_sentences), dtype=np.float32)
    y = np.zeros(num_rows, dtype=np.float32)
    w = np.zeros(num_rows, dtype=np.float32)

    for i, row in enumerate(rows):
        for j in range(no_sentences):
            X[i, j] = row.get(f"sent_{j}", 0)
        y[i] = row["similarity"]
        w[i] = max(float(row.get("weight", 1.0)), 1e-12)

    reg = LinearRegression()
    reg.fit(X, y, sample_weight=w)

    coef = np.asarray(reg.coef_, dtype=float).reshape(-1)
    scaler = MaxAbsScaler()
    norm_coef = scaler.fit_transform(coef.reshape(-1, 1)).reshape(-1)

    return coef, norm_coef, reg


def _loo_sentence_drops_batch(model, query, sentences, batch_size=32):
    """
    Score drop when removing each sentence from the selected sentence set.
    Positive value means the sentence supports the recommendation score.
    """
    n = len(sentences)
    if n == 0:
        return np.array([], dtype=float)

    full_text = " ".join(sentences).strip()
    reduced_texts = [
        " ".join(sentences[:i] + sentences[i + 1:]).strip()
        for i in range(n)
    ]

    texts = [full_text] + reduced_texts
    scores = model.predict_score_batch(query, texts, batch_size=batch_size)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if scores.size != n + 1:
        full_score = float(model.predict_score(query, sentences))
        drops = []
        for i in range(n):
            reduced = sentences[:i] + sentences[i + 1:]
            reduced_score = float(model.predict_score(query, reduced)) if reduced else 0.0
            drops.append(full_score - reduced_score)
        return np.asarray(drops, dtype=float)

    full_score = float(scores[0])
    return full_score - scores[1:]


def basic_sentence_split(text, min_sentence_chars=8):
    """Simple rule-based sentence splitter used by SmartSHAP-Sentence."""
    if text is None:
        return []

    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) >= min_sentence_chars]


# =========================================================
# 3. BASELINE / CLASSIC EXPLAINERS
# =========================================================
class BaselineExplainer:
    """Reference KernelSHAP baseline with a larger sample budget."""

    def __init__(self, model, sentences, query):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query

    def explain(self, n_samples=1000):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        def predict_fn(mask_matrix):
            outputs = []
            for mask in mask_matrix:
                selected = [
                    self.sentences[j]
                    for j, m in enumerate(mask)
                    if m > 0.5
                ]
                outputs.append(self.model.predict_score(self.query, selected))
            return np.asarray(outputs, dtype=float)

        explainer = shap.KernelExplainer(predict_fn, np.zeros((1, n)))
        vals = explainer.shap_values(
            np.ones((1, n)),
            nsamples=n_samples,
            silent=True,
        )

        return np.asarray(vals[0] if isinstance(vals, list) else vals, dtype=float).reshape(-1)


class VanillaKernelSHAP:
    """Lower-budget KernelSHAP baseline."""

    def __init__(self, model, sentences, query):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query

    def explain(self, n_samples=200):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        def predict_fn(mask_matrix):
            outputs = []
            for mask in mask_matrix:
                selected = [
                    self.sentences[j]
                    for j, m in enumerate(mask)
                    if m > 0.5
                ]
                outputs.append(self.model.predict_score(self.query, selected))
            return np.asarray(outputs, dtype=float)

        vals = shap.KernelExplainer(
            predict_fn,
            np.zeros((1, n)),
        ).shap_values(
            np.ones((1, n)),
            nsamples=n_samples,
            silent=True,
        )

        return np.asarray(vals[0] if isinstance(vals, list) else vals, dtype=float).reshape(-1)


class LIMEExplainer:
    def __init__(self, model, sentences, query):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.explainer = LimeTextExplainer(
            class_names=["Not Relevant", "Relevant"],
            char_level=False,
            split_expression=r" \| ",
        )

    def explain(self, num_samples=500):
        if not self.sentences:
            return np.array([], dtype=float)

        query_emb = self.model.encode_texts([self.query], batch_size=1, convert_to_numpy=True)
        query_emb = np.asarray(query_emb, dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)

        def classifier_fn(texts):
            if texts is None or len(texts) == 0:
                return np.empty((0, 2), dtype=float)

            doc_embs = self.model.encode_texts(texts, batch_size=32, convert_to_numpy=True)
            doc_embs = np.asarray(doc_embs, dtype=np.float32)
            if doc_embs.ndim == 1:
                doc_embs = doc_embs.reshape(1, -1)

            doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
            probs = (doc_embs @ query_emb.T).reshape(-1)
            probs = np.clip(probs, 0.0, 1.0)

            return np.vstack([1.0 - probs, probs]).T

        text_instance = " | ".join(self.sentences)

        exp = self.explainer.explain_instance(
            text_instance,
            classifier_fn,
            num_features=len(self.sentences),
            num_samples=num_samples,
        )

        dict_weights = dict(exp.as_list())
        return np.asarray(
            [dict_weights.get(sent, 0.0) for sent in self.sentences],
            dtype=float,
        )


class LeaveOneOutExplainer:
    def __init__(self, model, sentences, query):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query

    def explain(self):
        return _loo_sentence_drops_batch(
            self.model,
            self.query,
            self.sentences,
            batch_size=32,
        )


class BanzhafExplainer:
    def __init__(self, model, sentences, query, random_state=42):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.random_state = random_state

    def explain(self, n_samples=100, batch_size=16):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        rng = np.random.default_rng(self.random_state)
        num_masks = min(int(n_samples), 2 ** n if n < 10 else int(n_samples))
        num_masks = max(num_masks, n + 1)

        all_masks = rng.integers(0, 2, size=(num_masks, n))

        # Ensure basic anchors are present.
        for i in range(min(n, num_masks)):
            all_masks[i] = 0
            all_masks[i, i] = 1
        if num_masks > n:
            all_masks[n] = 1

        texts = [
            " ".join(self.sentences[j] for j in range(n) if mask[j]).strip()
            for mask in all_masks
        ]

        scores = self.model.predict_score_batch(self.query, texts, batch_size=batch_size)
        scores = np.asarray(scores, dtype=float).reshape(-1)

        reg = LinearRegression()
        reg.fit(all_masks, scores)

        return _max_abs_normalize(reg.coef_)


class RISEExplainer:
    def __init__(self, model, sentences, query, random_state=42):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.random_state = random_state

    def explain(self, n_samples=500, batch_size=32):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        rng = np.random.default_rng(self.random_state)
        masks = rng.random((int(n_samples), n)) > 0.5

        # Avoid empty masks.
        empty_rows = np.where(~masks.any(axis=1))[0]
        for row in empty_rows:
            masks[row, rng.integers(0, n)] = True

        texts = [
            " ".join(self.sentences[j] for j in range(n) if mask[j]).strip()
            for mask in masks
        ]

        scores = self.model.predict_score_batch(self.query, texts, batch_size=batch_size)
        scores = np.asarray(scores, dtype=float).reshape(-1)

        weights = (masks.astype(float) * scores[:, None]).sum(axis=0)
        counts = masks.astype(float).sum(axis=0)

        return np.divide(
            weights,
            counts,
            out=np.zeros_like(weights, dtype=float),
            where=counts > 0,
        )


class RandomExplainer:
    def __init__(self, sentences, random_state=42):
        self.sentences = list(sentences) if sentences is not None else []
        self.random_state = random_state

    def explain(self):
        rng = np.random.default_rng(self.random_state)
        return rng.random(len(self.sentences))


# =========================================================
# 4. SMARTSHAP EXPLAINERS
# =========================================================
class SmartShapExplainer:
    """
    SmartSHAP core for already segmented sentence inputs.

    Components:
    - structured coalition sampling;
    - batch bi-encoder scoring;
    - kernel-weighted linear surrogate;
    - optional LOO calibration for faithfulness-oriented sentence ranking.

    If use_loo_calibration=True:
        final = shap_weight * SHAP_component + (1 - shap_weight) * LOO_component
    """

    def __init__(
        self,
        model,
        sentences,
        query,
        max_samples=1000,
        batch_size=32,
        random_state=42,
        use_loo_calibration=True,
        shap_weight=0.65,
    ):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.max_samples = int(max_samples)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.use_loo_calibration = bool(use_loo_calibration)
        self.shap_weight = float(np.clip(shap_weight, 0.0, 1.0))

    def explain(self, n_samples=None, return_raw=False):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        actual_samples = int(n_samples) if n_samples is not None else self.max_samples
        combinations = generate_shap_samples(n, actual_samples, self.random_state)

        texts = [
            " ".join(self.sentences[i] for i in comb).strip()
            for comb in combinations
        ]

        query_emb = self.model.encode_texts(
            [self.query],
            batch_size=1,
            convert_to_numpy=True,
        )[0]

        doc_embs = self.model.encode_texts(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        doc_embs = np.asarray(doc_embs, dtype=np.float32)
        if doc_embs.ndim == 1:
            doc_embs = doc_embs.reshape(1, -1)

        rows = []
        for comb, emb in zip(combinations, doc_embs):
            row = {f"sent_{i}": 0 for i in range(n)}
            for i in comb:
                row[f"sent_{i}"] = 1

            row.update({
                "similarity": _safe_cosine_from_embeddings(query_emb, emb),
                "weight": _kernel_shap_weight(n, len(comb)),
                "subset_size": len(comb),
            })
            rows.append(row)

        coef, shap_norm, reg = fit_shap_model_from_rows(rows, n)
        shap_component = _max_abs_normalize(shap_norm)

        loo_raw = np.zeros(n, dtype=float)
        loo_component = np.zeros(n, dtype=float)

        if self.use_loo_calibration and n > 1:
            loo_raw = _loo_sentence_drops_batch(
                self.model,
                self.query,
                self.sentences,
                batch_size=self.batch_size,
            )
            loo_component = _max_abs_normalize(loo_raw)
            final_vals = (
                self.shap_weight * shap_component
                + (1.0 - self.shap_weight) * loo_component
            )
        else:
            final_vals = shap_component.copy()

        final_vals = _max_abs_normalize(final_vals)

        if return_raw:
            return {
                "coef": coef,
                "normalized_coef": final_vals,
                "shap_component": shap_component,
                "loo_raw": loo_raw,
                "loo_component": loo_component,
                "rows": rows,
                "reg_model": reg,
                "num_sentences": n,
                "num_samples": len(rows),
                "use_loo_calibration": self.use_loo_calibration,
                "shap_weight": self.shap_weight,
            }

        return final_vals


class SmartShapSentenceExplainer:
    """
    SmartSHAP-Sentence for raw document text.

    Workflow:
    - split the recommended document into sentence units;
    - optionally keep only max_segments sentences;
    - use query-aware sentence preselection when the document is long;
    - run the SmartSHAP attribution core over selected sentence units.
    """

    def __init__(
        self,
        model,
        document_text,
        query,
        max_segments=12,
        max_samples=200,
        batch_size=32,
        random_state=42,
        min_sentence_chars=8,
        selection_mode="query_topk",
        use_loo_calibration=True,
        shap_weight=0.65,
    ):
        self.model = model
        self.document_text = document_text
        self.query = query
        self.max_segments = max_segments
        self.max_samples = int(max_samples)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.min_sentence_chars = min_sentence_chars
        self.selection_mode = selection_mode
        self.use_loo_calibration = bool(use_loo_calibration)
        self.shap_weight = float(np.clip(shap_weight, 0.0, 1.0))

        self.all_sentences = basic_sentence_split(
            document_text,
            min_sentence_chars=self.min_sentence_chars,
        )
        (
            self.sentences,
            self.selected_sentence_indices,
            self.sentence_selection_scores,
        ) = self._select_sentences(self.all_sentences)

    def _select_sentences(self, sentences):
        sentences = list(sentences)
        n = len(sentences)

        if n == 0:
            return [], [], np.array([], dtype=float)

        if self.max_segments is None or self.max_segments <= 0 or n <= self.max_segments:
            return sentences, list(range(n)), np.full(n, np.nan, dtype=float)

        if self.selection_mode == "first":
            keep_idx = list(range(int(self.max_segments)))
            return [sentences[i] for i in keep_idx], keep_idx, np.full(n, np.nan, dtype=float)

        selection_scores = self.model.predict_score_batch(
            self.query,
            sentences,
            batch_size=self.batch_size,
        )
        selection_scores = np.asarray(selection_scores, dtype=float).reshape(-1)

        if selection_scores.size != n or not np.all(np.isfinite(selection_scores)):
            keep_idx = list(range(int(self.max_segments)))
            return [sentences[i] for i in keep_idx], keep_idx, selection_scores

        top_idx = np.argsort(selection_scores)[::-1][: int(self.max_segments)]
        keep_idx = sorted(int(i) for i in top_idx)

        return [sentences[i] for i in keep_idx], keep_idx, selection_scores

    def split_into_sentences(self, text):
        sentences = basic_sentence_split(text, min_sentence_chars=self.min_sentence_chars)
        selected, _, _ = self._select_sentences(sentences)
        return selected

    def generate_samples(self, n, max_samples):
        return generate_shap_samples(
            no_sentences=n,
            max_samples=max_samples,
            random_state=self.random_state,
        )

    def explain(self, n_samples=None, return_raw=False):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        actual_samples = int(n_samples) if n_samples is not None else self.max_samples

        core = SmartShapExplainer(
            model=self.model,
            sentences=self.sentences,
            query=self.query,
            max_samples=actual_samples,
            batch_size=self.batch_size,
            random_state=self.random_state,
            use_loo_calibration=self.use_loo_calibration,
            shap_weight=self.shap_weight,
        )

        result = core.explain(n_samples=actual_samples, return_raw=return_raw)

        if not return_raw:
            return result

        result.update({
            "sentences": self.sentences,
            "all_sentences": self.all_sentences,
            "selected_sentence_indices": self.selected_sentence_indices,
            "sentence_selection_scores": self.sentence_selection_scores,
            "num_all_sentences": len(self.all_sentences),
            "selection_mode": self.selection_mode,
        })

        return result


class SmartShapLOOEnsemble:
    """
    SmartSHAP + Leave-One-Out ensemble.

    final = alpha * SmartSHAP + (1 - alpha) * LOO
    """

    def __init__(self, model, sentences, query, alpha=0.7):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def explain(self, n_samples=200):
        if not self.sentences:
            return np.array([], dtype=float)

        smart_vals = SmartShapExplainer(
            self.model,
            self.sentences,
            self.query,
            use_loo_calibration=False,
        ).explain(n_samples=n_samples)

        loo_vals = LeaveOneOutExplainer(
            self.model,
            self.sentences,
            self.query,
        ).explain()

        smart_norm = _l1_normalize(smart_vals)
        loo_norm = _l1_normalize(loo_vals)

        return self.alpha * smart_norm + (1.0 - self.alpha) * loo_norm


# Backward-compatible name if older main.py still imports this class.
LOO_SHAP_Ensemble = SmartShapLOOEnsemble


# =========================================================
# 5. ADAPTIVE / SCREENING SMARTSHAP VARIANTS
# =========================================================
class AdaptiveSmartShapLOO:
    """
    Adaptive SmartSHAP-LOO.

    For short documents, the method gives higher weight to SmartSHAP.
    For longer documents, it gradually increases the role of LOO score drops.
    """

    def __init__(self, model, sentences, query, shap_weight_slope=0.3, pivot=10):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.shap_weight_slope = float(shap_weight_slope)
        self.pivot = float(pivot)

    def _get_adaptive_alpha(self, n):
        return 1.0 / (1.0 + np.exp(self.shap_weight_slope * (n - self.pivot)))

    def _get_smartshap_values(self, n_samples=200):
        return SmartShapExplainer(
            self.model,
            self.sentences,
            self.query,
            use_loo_calibration=False,
        ).explain(n_samples=n_samples)

    def explain(self, n_samples=200):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        alpha = self._get_adaptive_alpha(n)

        smart_vals = np.asarray(
            self._get_smartshap_values(n_samples=n_samples),
            dtype=float,
        )
        loo_vals = np.asarray(
            LeaveOneOutExplainer(self.model, self.sentences, self.query).explain(),
            dtype=float,
        )

        smart_norm = _l1_normalize(smart_vals)
        loo_norm = _l1_normalize(loo_vals)

        return alpha * smart_norm + (1.0 - alpha) * loo_norm


class ScreenerSmartShap:
    """
    Two-stage SmartSHAP variant.

    Stage 1: LOO screening.
    Stage 2: SmartSHAP refinement over the top-k selected sentences.
    """

    def __init__(self, model, sentences, query, top_k=5):
        self.model = model
        self.sentences = list(sentences) if sentences is not None else []
        self.query = query
        self.top_k = int(top_k)

    def _refine(self, sub_sentences, n_samples=200):
        return SmartShapExplainer(
            self.model,
            sub_sentences,
            self.query,
            use_loo_calibration=False,
        ).explain(n_samples=n_samples)

    def explain(self, n_samples=200):
        n = len(self.sentences)
        if n == 0:
            return np.array([], dtype=float)

        if n <= self.top_k:
            return self._refine(self.sentences, n_samples=n_samples)

        loo_scores = LeaveOneOutExplainer(
            self.model,
            self.sentences,
            self.query,
        ).explain()

        top_idx = np.argsort(np.abs(loo_scores))[-self.top_k:]
        top_idx = np.sort(top_idx)

        sub_sentences = [self.sentences[i] for i in top_idx]
        sub_vals = self._refine(sub_sentences, n_samples=n_samples)

        full_vals = np.zeros(n, dtype=float)
        for i, global_idx in enumerate(top_idx):
            full_vals[global_idx] = sub_vals[i]

        return full_vals
