"""Real-time CLV inference: XGBoost RFM baseline, trained artifact-dan yüklənir.

BTYD (BG/NBD + Gamma-Gamma) burada BİLƏRƏKDƏN istifadə edilmir: onun `T` parametri
(ilk alışdan bəri keçən müddət) təlim zamanı SABİT bir `observation_period_end`
kəsişməsinə görə hesablanıb (bax: clv/train.py, clv/btyd.py). Canlı sorğuda bunu
düzgün yeniləmək hər dəfə tam tranzaksiya tarixçəsini yenidən emal etməyi tələb
edir — <200ms tək-müştəri endpoint-i üçün uyğun deyil. Ona görə BTYD batch/offline
skorlama üçün saxlanılır, real-time XGBoost baseline istifadə edir (bu, RFM
dəyərlərini birbaşa qəbul etdiyi üçün stateless işləyir).
"""
from __future__ import annotations

import joblib
import pandas as pd


def predict_clv(
    recency: float,
    frequency: float,
    monetary: float,
    artifact_path: str = "models/clv_model.joblib",
) -> float:
    artifact = joblib.load(artifact_path)
    model = artifact["xgb_baseline"]
    feature_cols = artifact["feature_cols"]  # ["recency", "frequency", "monetary"]

    row = pd.DataFrame(
        [{"recency": recency, "frequency": frequency, "monetary": monetary}]
    )[feature_cols]

    pred = float(model.predict(row)[0])
    # XGBoost regression mənfi qiymət qaytara bilər (CLV mənasız olardı) —
    # priority score-un ValueError atmaması üçün 0-da kəsirik.
    return max(0.0, pred)