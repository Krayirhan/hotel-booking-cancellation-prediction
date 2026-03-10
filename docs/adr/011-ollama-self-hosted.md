# ADR-002: Kendi Kendine Barındırılan Ollama (self-hosted LLM + Embeddings)

**Durum:** Kabul Edildi  
**Tarih:** 2026-02-17  
**Karar Vericiler:** Proje Ekibi  

---

## Bağlam

Chat/RAG sisteminin bir LLM (otel politikası soruları için) ve gömme modeline (pgvector benzerlik araması için `nomic-embed-text`) ihtiyacı vardır. Harici LLM SaaS sağlayıcıları (OpenAI, Anthropic) ücretli API anahtarlarına, ağ gecikmesine ve üçüncü taraf veri işlemeye bağımlılık gerektirir.

---

## Karar

LLM çıkarımı ve gömme üretimi için Ollama'yı **kendi sunucumuzda çalıştırıyoruz**. Varsayılan model `qwen2.5:7b`; gömme modeli `nomic-embed-text` (768 boyutlu vektörler).

```
stack: Ollama (Docker image veya native daemon)
       ↓ HTTP (localhost:11434 veya OLLAMA_BASE_URL env)
src/chat/ollama_client.py — async wrapper
       ↓
src/chat/pipeline/ — intent → context → prompt → validate aşamaları
       ↓
pgvector (PostgreSQL) — bilgi parçaları 768 boyutlu HNSW indeksinde
```

Üretimde `OLLAMA_BASE_URL` ortam değişkeni Kubernetes içindeki Ollama pod'una işaret eder. Docker geliştirme yığınında `host.docker.internal:11434` kullanılır.

---

## Sonuçlar

**Olumlu:**
- Sıfır LLM API maliyeti; misafir verisi asla dış sisteme gitmez.
- Model seçimi çalışma zamanında yapılandırılabilir (OLLAMA_MODEL env).
- Senkron ve asenkron çağrı ('embed_sync', 'generate_async') herhangi bir FastAPI handler ile uyumlu.

**Olumsuz/Dengeler:**
- Üretim kümesinde Ollama pod'u için GPU kaynağı veya büyük CPU tahsisi gerekir.
- Model soğuk başlatması (ilk yükleme) ek gecikme yaratır; sağlık kontrolleri buna göre ayarlanmıştır.
- Ollama erişilemez hale gelirse, chat sistemi degrade moda düşer (hata mesajı döndürür, süreci kilitlemez).

---

## Reddedilen Alternatifler

- **OpenAI / GPT-4o API**: Veri mahremiyeti riski; abonelik maliyeti; internet bağlantısı gereksinimi.
- **Hugging Face Inference API**: Ücretsiz katmanda hız sınırlaması; yerinde denetim yoktur.
- **llama.cpp direkt entegrasyon**: Daha az gömülü yönetim API'si; model idaresi daha yüksek mühendislik maliyeti gerektirir.
