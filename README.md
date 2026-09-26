 # Customer Analytics Suite

End-to-end customer analytics system: **Churn Prediction**, **Next Best Product/Offer (NBP/NBO)**, and **Customer Lifetime Value (CLV)** — unified through a single `priority_score`.

## Quick start

```bash
# 1. clone & enter
git clone <your-repo-url> customer-analytics-suite
cd customer-analytics-suite

# 2. create virtual env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. install
make install

# 4. put raw datasets into data/raw/ (see Data sources below), then train
make train-churn

# 5. run the API
make api
# -> http://127.0.0.1:8000/docs

# 6. run the dashboard
make dashboard
```

## Project structure

See `src/cas/{churn,nbo,clv,common,api}` — each model module implements a consistent
interface (data → features → train/fit → predict → explain) so the API and dashboard
can call all three uniformly.

## Data sources
| Module | Dataset | Source |
|---|---|---|
| Churn | Telco Customer Churn | Kaggle (IBM sample) |
| NBO | Bank Marketing | UCI ML Repository |
| CLV | Online Retail II | UCI ML Repository |

## Priority score

```
priority_score = churn_probability × CLV × (1 + best_offer_propensity)
```

## Results (fill in once trained)
- Churn model: ROC-AUC —, PR-AUC —, top-decile lift —
- NBO model: Precision@3 —, hit rate —
- CLV model: Spearman ρ —
### Məlum məhdudiyyət: CLV real-time baseline-ın nöqtəvi proqnozları

`/predict/clv` endpoint-i XGBoost RFM baseline istifadə edir (bax bölmə "Niyə BG/NBD, niyə
XGBoost real-time üçün"). Bu model 90 günlük holdout dövründəki REAL xərcə görə təlim
olunub — hədəf dəyişəni kəskin sağa-əyridir (çox müştəri həmin dövrdə heç nə xərcləmir).
Nəticədə:
- Tək-tək proqnozlar mənfi ola bilər (sıfır ətrafında reqressiya səs-küyü) — bunlar `predict.py`-də
  0-a kəsilir.
- Model orta Spearman ρ = 0.484 ilə işləyir (BTYD-nin 0.582-i ilə müqayisədə zəif) — yəni
  ÜMUMİ sıralamanı qismən tutur, amma tək-tək müştərilər arasında ciddi qeyri-monotonluq
  gözlənilir.
- Bu səbəbdən API-nin çıxışı mütləq dəqiq dollar məbləği kimi deyil, NİSBİ prioritetləşdirmə
  siqnalı kimi oxunmalıdır (məhz buna görə priority_score düsturunda CLV çarpan kimi istifadə
  olunur — sıralama üçün, tək başına "həqiqət" kimi deyil).

**Gələcək yaxşılaşdırma:** `log1p(actual_spend)` üzərində təlim etmək sağa-əyriliyi azaldıb
mənfi proqnozları demək olar ki, aradan qaldırardı — v2 üçün planlaşdırılıb.

## Tests & CI
```bash
make test    # pytest + coverage
make lint    # ruff
```
GitHub Actions runs both on every push/PR (`.github/workflows/ci.yml`).
