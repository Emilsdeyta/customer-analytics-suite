"""
Priority score: churn, CLV və NBO modullarının nəticələrini
vahid, biznes üçün oxunaqlı bir metrikdə birləşdirir.

    priority_score = churn_probability * clv * (1 + best_offer_propensity)

Dizayn qərarları:
- compute_priority_score() PURE funksiyadır: heç bir model, heç bir data
  mənbəyi ilə bağlı deyil. Yalnız 3 float qəbul edir, 1 float qaytarır.
- churn və clv modulları "hazır feature ver" məntiqi ilə qurulub
  (customer_id-dən daxili axtarış YOXDUR — çağıran tərəf feature/RFM
  dəyərlərini özü ötürür). Yalnız NBO modulu customer_id-yə görə daxili
  axtarış edir (artefaktın içindəki interaction matrix/rules vasitəsilə).
  Bu 3 fərqli interfeys eyni orkestrasiya funksiyasında BİRLƏŞDİRİLİR.
- best_offer_propensity default 0.0-dır: NBO boş siyahı qaytarsa,
  multiplier 1.0 olur, skor sıfırlanmır.
- churn_customer_id / retail_customer_id İKİ AYRI dataset-in müştəri
  ID-ləridir (Telco vs Online Retail II) — bu portfolio layihəsində
  sintetik cütləşdirmədir. Production-da eyni customer_id olardı.
"""

from __future__ import annotations

from dataclasses import dataclass

from cas.churn.predict import predict_one
from cas.clv.predict import predict_clv
from cas.nbo.predict import recommend_for_customer


@dataclass(frozen=True)
class PriorityScoreResult:
    priority_score: float
    churn_probability: float
    clv: float
    best_offer_propensity: float
    best_offer_product: str | None = None


def compute_priority_score(
    churn_probability: float,
    clv: float,
    best_offer_propensity: float = 0.0,
) -> float:
    """Pure scoring funksiyası. Data mənbəyindən asılı deyil."""
    if not (0.0 <= churn_probability <= 1.0):
        raise ValueError(
            f"churn_probability [0,1] aralığında olmalıdır, alındı: {churn_probability}"
        )
    if clv < 0:
        raise ValueError(f"clv mənfi ola bilməz, alındı: {clv}")
    if not (0.0 <= best_offer_propensity <= 1.0):
        raise ValueError(
            f"best_offer_propensity [0,1] aralığında olmalıdır, alındı: {best_offer_propensity}"
        )

    return churn_probability * clv * (1.0 + best_offer_propensity)


def score_customer_pair(
    churn_customer_id: str,
    churn_features: dict,
    retail_customer_id: str,
    retail_recency: float,
    retail_frequency: float,
    retail_monetary: float,
    churn_artifact_path: str = "models/churn_model.joblib",
    clv_artifact_path: str = "models/clv_model.joblib",
    nbo_artifact_path: str = "models/nbo_model.joblib",
) -> PriorityScoreResult:
    """Orkestrasiya qatı: 3 modulu real interfeyslərinə uyğun çağırır."""
    churn_prob, _ = predict_one(churn_features, artifact_path=churn_artifact_path)

    clv_value = predict_clv(
        recency=retail_recency,
        frequency=retail_frequency,
        monetary=retail_monetary,
        artifact_path=clv_artifact_path,
    )

    offers = recommend_for_customer(retail_customer_id, artifact_path=nbo_artifact_path)
    if offers:
        # offers artıq hybrid_rank() tərəfindən combined_score-a görə
        # (propensity * association-rule lift) azalan sırada verilib —
        # deməli offers[0] elə ƏN YAXŞI hibrid təklifdir, max() ilə
        # yenidən axtarmağa ehtiyac yoxdur (əksinə, max(propensity) səhv
        # nəticə verərdi, çünki lift-in töhfəsini nəzərə almazdı).
        #
        # Multiplier üçün isə xam `propensity` (classifier ehtimalı,
        # [0,1]-də) istifadə olunur, `combined_score` yox — çünki
        # combined_score lift > 1 olduqda 1.0-ı keçə bilər və
        # (1 + best_offer_propensity) düsturunu qeyri-mütənasib şişirdərdi.
        best_offer = offers[0]
        best_offer_propensity = best_offer["propensity"]
        best_offer_product = best_offer["product"]
    else:
        best_offer_propensity = 0.0
        best_offer_product = None

    score = compute_priority_score(
        churn_probability=churn_prob,
        clv=clv_value,
        best_offer_propensity=best_offer_propensity,
    )

    return PriorityScoreResult(
        priority_score=score,
        churn_probability=churn_prob,
        clv=clv_value,
        best_offer_propensity=best_offer_propensity,
        best_offer_product=best_offer_product,
    )