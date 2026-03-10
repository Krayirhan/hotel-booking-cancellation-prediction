# ADR-001: Versiyonlu API Tasarımı (v1 / v2)

**Durum:** Kabul Edildi  
**Tarih:** 2026-02-17  
**Karar Vericiler:** Proje Ekibi  

---

## Bağlam

Üretim sistemleri, API değişikliklerinden etkilenen birden fazla istemci (frontend dashboard, k6 yük testi, Helm values.yaml içindeki cron job'lar) içerir. Mevcut `/predict_proba` ve `/decide` endpoint'leri üzerinde geriye dönük uyumsuz değişiklikler yapılması gerekiyordu (özellikle v2 için canlı açıklanabilirlik desteği).

---

## Karar

Üretim API'si **iki sürüm** olarak dağıtılır:

| Sürüm | Önek | Sorumluluk |
|-------|------|------------|
| V1 | `/v1/` | Geriye dönük uyumlu; `predict_proba`, `decide`, `reload` |
| V2 | `/v2/` | Zenginleştirilmiş yanıtlar; `/v2/explain/{run_id}` — SHAP/permütasyon önem raporları |

Her iki sürüm de `src/api_shared.py` üzerinden `ServingState`, `RecordsPayload` ve çalıştırma yardımcılarını paylaşır.  
`src/api.py` tek bir FastAPI uygulaması oluşturur ve her iki router'ı `include_router` ile ekler.

---

## Sonuçlar

**Olumlu:**
- İstemciler bozulmadan çalışmaya devam eder; v2 özellikleri ayrı bir yolda sunulur.
- `x-api-key` ve rate-limit middleware'i her iki sürüm için de paylaşımlıdır.
- OpenAPI şeması her sürüm için ayrı label içerir; dokümantasyon erişilebilir kalır.

**Olumsuz/Dengeler:**
- Ağırlıklı olarak birleşik test gerçekleştirilir; her sürüm kendi sözleşmesine bağımlıdır.
- Sürüm göçlerini geriye dönük uyumlu tutmak ek özen gerektirir.
- Bir an için üç yol bulunmaktadır: kök (`/predict_proba`), `/v1/`, `/v2/` — uzun vadede kök yol kaldırılacak.

---

## Reddedilen Alternatifler

- **URL parametresi sorgu dizesi** (`?version=2`): REST uygulamasında yaygın değil; önbellek geçersizleştirmeyi karıştırır.
- **Başlık bazlı sürümleme** (`Accept: application/vnd.ds.v2+json`): İstemci uygulaması daha karmaşık; tarayıcı araçlarında hata ayıklaması zor.
- **Tek sürümlü API**: v2 özellikleri için geriye dönük uyumsuz değişiklikler yapılmasını engeller.
