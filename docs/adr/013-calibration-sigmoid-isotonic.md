# ADR-004: İki Aşamalı Kalibrasyon (Sigmoid + Isotonic)

**Durum:** Kabul Edildi  
**Tarih:** 2026-02-17  
**Karar Vericiler:** Proje Ekibi  

---

## Bağlam

Ham ML modelleri (XGBoost, LightGBM, vs.) sınıf olasılıklarını doğru biçimde ifade etmeyebilir — özellikle dengesiz veri setlerinde. Otel iptali karar politikası, gerçek iptal oranlarıyla uyumlu güvenilir olasılık tahminleri gerektirir (örn. %60 tahmini gerçekten orada %60 olmalı). Yanlış kalibre olmuş modeller düşük maliyetli kararlar için eşiğin ayarlanmasını zorlaştırır.

---

## Karar

Her çekirdek model için **iki kalibratör** eğitilip değerlendirilir:

| Kalibratör | Ne zaman iyi çalışır? |
|------------|----------------------|
| **Sigmoid** (Platt Scaling) | Model çıktısı sigmoid şekline benziyorsa; daha az veriyle eğitimde daha stabil |
| **Isotonic Regression** | Kalibrasyon hatası monoton olmadığında; daha fazla veri mevcut olduğunda |

Daha iyi Brier skoru veya log-kaybı alan kalibratör, seçim için şampiyon modele eklenir. Seçim `reports/metrics/{run_id}/calibration_metrics.json` dosyasına kaydedilir.

```
train() → raw model (XGBoost, LightGBM, LR, CatBoost, vb.)
       ↓
calibrate() → CalibratedClassifierCV(method='sigmoid')
            → CalibratedClassifierCV(method='isotonic')
       ↓
pick_best_calibrator() → Brier skoru karşılaştırması
       ↓
serialize → models/baseline_calibrated_sigmoid.joblib
           models/baseline_calibrated_isotonic.joblib
```

---

## Sonuçlar

**Olumlu:**
- Gerçekçi olasılıklar: `%Pt tahmin ≈ gerçek iptal oranı` her eşik değerinde.
- Karar politikası için maliyet-duyarlı eşikleme (beklenen kâr maks.) güvenilir hale gelir.
- Her iki kalibratör da model kayıt defterinde günlüğe kaydedilir; gerekirse el ile geçiş yapılabilir.

**Olumsuz/Dengeler:**
- Kalibrasyonu eğitmek için çapraz doğrulamadan ayrı bir tutma seti (`cal fold`) gerekir.
- İzotonik regresyon monotonluğu zorlar; bazı model tiplerinde sigmoid daha genellenebilirdir.
- Çok küçük veri setlerinde (`<500 örnek`) izotonik yüksek varyans gösterir — sigmoid tercih edilir.

---

## Reddedilen Alternatifler

- **Kalibrasyon yok**: Ham olasılıklar güvenilmez; eşik seçimi anlamsızlaşır.
- **Temperature Scaling**: Sinir ağları için iyi çalışır ancak sklearn topluluğunda doğrudan destek bulunmadığı için uygulaması daha karmaşıktır.
- **Beta Kalibrasyon**: Gür kayıp fonksiyonu; pratik iyileştirme marjinal; fazladan bağımlılık gerektirir.
