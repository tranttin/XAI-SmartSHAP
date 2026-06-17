import numpy as np
from scipy.stats import spearmanr

class XAIEvaluator:
    """Module đánh giá định lượng XAI."""

    @staticmethod
    def evaluate_faithfulness(query, sentences, shap_values, model, k=1, use_abs=False):
        """
        Đo Comprehensiveness và Sufficiency ở mức top-k.

        - Comprehensiveness = full_score - score_without_topk
          Càng lớn càng cho thấy top-k thực sự quan trọng.

        - Sufficiency = full_score - score_with_only_topk
          Càng nhỏ càng tốt, nghĩa là chỉ top-k đã đủ giữ lại phần lớn tín hiệu.
        """
        if len(sentences) == 0:
            return {"comprehensiveness": 0.0, "sufficiency": 0.0}

        full_score = model.predict_score(query, sentences)
        actual_k = min(k, len(sentences))

        ranking_values = np.abs(shap_values) if use_abs else shap_values
        top_k_idx = np.argsort(ranking_values)[-actual_k:]

        reduced_sents = [s for i, s in enumerate(sentences) if i not in top_k_idx]
        top_sents = [sentences[i] for i in top_k_idx]

        score_without_topk = model.predict_score(query, reduced_sents) if reduced_sents else 0.0
        score_with_only_topk = model.predict_score(query, top_sents) if top_sents else 0.0

        return {
            "comprehensiveness": float(full_score - score_without_topk),
            "sufficiency": float(full_score - score_with_only_topk)
        }

    @staticmethod
    def evaluate_rank_agreement(shap_a, shap_b):
        """
        Đo Spearman rank correlation giữa 2 vector explanation.
        Dùng để so với high-budget baseline SHAP.
        """
        if len(shap_a) != len(shap_b) or len(shap_a) < 2:
            return 0.0

        corr, _ = spearmanr(shap_a, shap_b)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def evaluate_error(shap_a, shap_b):
        """
        Đo sai số trực tiếp giữa method và baseline:
        - MAE
        - RMSE
        """
        if len(shap_a) != len(shap_b) or len(shap_a) == 0:
            return {"mae": 0.0, "rmse": 0.0}

        shap_a = np.array(shap_a, dtype=float)
        shap_b = np.array(shap_b, dtype=float)
        diff = shap_a - shap_b

        return {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2)))
        }

    @staticmethod
    def evaluate_stability(shap_runs):
        """
        Đo stability thật sự qua nhiều lần chạy:
        tính trung bình Spearman correlation giữa mọi cặp runs.

        Input:
            shap_runs: list các vector explanation từ nhiều lần chạy

        Output:
            stability score trong [-1, 1], càng cao càng ổn định
        """
        if shap_runs is None or len(shap_runs) < 2:
            return 0.0

        corrs = []
        for i in range(len(shap_runs)):
            for j in range(i + 1, len(shap_runs)):
                a, b = shap_runs[i], shap_runs[j]
                if len(a) != len(b) or len(a) < 2:
                    continue
                corr, _ = spearmanr(a, b)
                if not np.isnan(corr):
                    corrs.append(float(corr))

        return float(np.mean(corrs)) if corrs else 0.0