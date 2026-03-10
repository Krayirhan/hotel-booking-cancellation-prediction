# ADR-003: PostgreSQL + Redis Çift Katman Depolama

**Durum:** Kabul Edildi  
**Tarih:** 2026-02-17  
**Karar Vericiler:** Proje Ekibi  

---

## Bağlam

Sistemin birden fazla depolama gereksinimi bulunmaktadır:

| Gereksinim | Özellikler |
|------------|------------|
| Dashboard metrik geçmişi | İlişkisel; sorgulanabilir; veri uzun süre saklanır |
| pgvector RAG bilgi tabanı | Vektör benzerlik araması; HNSW indeksi |
| Misafir profilleri | İlişkisel; CRUD; arama ve sayfalama |
| Auth token'ları | TTL-tabanlı; hızlı arama; çoklu worker |
| Rate limit bucket'ları | Atomik sayaç; TTL; dağıtık worker |
| Dashboard önbelleği | 45 saniyelik TTL; dosya taramasını azalt |

---

## Karar

**PostgreSQL 16 + pgvector** kalıcı ilişkisel ve vektör depolamasını üstlenir.  
**Redis 7** geçici, düşük gecikmeli ihtiyaçları karşılar.

```
PostgreSQL:
  - experiment_runs, model_metrics   (DashboardStore)
  - users                            (UserStore)
  - hotel_guests                     (GuestStore)
  - knowledge_chunks + vector(768)   (KnowledgeDbStore)

Redis:
  - ds:auth:tok:{token}              (auth token → TTL)
  - ds:auth:usr:{username}           (kullanıcı başına token seti)
  - ds:dash:overview:{run_id}        (dashboard önbelleği → 45 sn TTL)
  - ds:rl:{client}                   (rate limit bucket'ları → 60 sn TTL)
```

Her iki istemci de degrade modda çalışır: Redis kullanılamıyorsa auth bellekte çalışır, rate limit token'larına geri döner, dashboard önbelleği atlanır.

---

## Sonuçlar

**Olumlu:**
- Kalıcı veriler tam ACID güvencesiyle PostgreSQL'de korunur.
- pgvector, ayrı bir vektör veritabanı gerektirmeden yerleşik çalışır.
- Redis her zaman isteğe bağlıdır; degrade mod beklenmedik ölümlere karşı koruma sağlar.

**Olumsuz/Dengeler:**
- İki bağımlılık → iki servis konteyner'ı; yerel geliştirme biraz daha ağır.
- Çoklu worker ortamında (WEB_CONCURRENCY > 1) Redis olmadan rate limiting gerçek anlamda dağıtık değildir.

---

## Reddedilen Alternatifler

- **Yalnızca SQLite**: Üretimde çoklu worker ile dosya kilitleme sorunları; pgvector desteği yok.
- **Yalnızca PostgreSQL (önbellek de)**: SKIP LOCKED ile kuyruğa tabi SQL satırları üzerinden rate limiting garip performans gösterir; auth token aramalarına ek yük.
- **MongoDB**: Proje birincil kullanım durumu tablo örüntüleri; ilişkisel join'ler daha uygun.
