# Cost Analysis: 1000 Requests/Day

**GPU:** RTX 4090 @ $1.12/hr ($0.00031/sec)
**Target:** 1000 requests/day
**Generation time:** 30 seconds/request (after optimization)

---

## Scenario 1: Pure Pay-Per-Use (min_workers=0, idle_timeout=600)

### Calculation

```
1000 req/day × 30 sec/req = 30,000 sec/day = 8.33 hours/day

Monthly GPU time:
8.33 hrs/day × 30 days = 250 hours/month

Cost:
250 hours × $1.12/hr = $280/month
```

### BUT: Idle timeout adds overhead

Если запросы не постоянные, worker живет дольше:

```
Average request interval: 86 seconds (24hrs / 1000 req)
Idle timeout: 600 seconds (10 min)

Если запросы идут группами (burst):
- Утро: 300 req за 2 часа
- День: 400 req за 4 часа  
- Вечер: 300 req за 2 часа
- Ночь: тишина (worker спит)

Active time: ~10-12 hours/day (включая idle)
10 hrs × 30 days × $1.12 = $336/month
```

**Realistic cost: $280-400/month**

---

## Scenario 2: Always-On (min_workers=1)

```
Cost: $806/month (24/7)

При 1000 req/day:
- Worker ВСЕГДА готов (0 cold starts)
- Latency: стабильно 20-30 сек
- Predictable cost
- 19% времени работает, 81% idle (но вы платите за 100%)

Cost per request: $806 / 30,000 = $0.027/req
```

**Вывод:** Переплата за стабильность

---

## Scenario 3: Hybrid (min=1, но дешевле GPU для base)

Проблема: У вас RTX 4090 @ $1.12/hr - это ДОРОГО для always-on.

Стандартная цена RTX 4090:
- RunPod Secure Cloud: $0.39-0.44/hr
- RunPod Community Cloud: $0.30-0.35/hr

**Ваша цена $1.12/hr - это premium (возможно On-Demand или специальный регион)**

### Альтернатива: Найти дешевле RTX 4090

```
Если найти RTX 4090 @ $0.40/hr:

Always-on cost: $0.40 × 24 × 30 = $288/month
vs текущая: $806/month

ЭКОНОМИЯ: $518/month! (64%)
```

---

## Scenario 4: Smart Scaling (min=0, max=3)

При 1000 req/day возможны пики:

```
Normal load: 20-30 req/hour (1 worker)
Peak load: 100+ req/hour (нужно 2-3 workers)

Configuration:
min_workers: 0
max_workers: 3
idle_timeout: 600
scale_up_threshold: 5 requests in queue
```

### Cost calculation

```
Если распределение:
- 80% времени: 1 worker active
- 15% времени: 2 workers active  
- 5% времени: 3 workers active

Average workers: 1.25

Cost:
1.25 workers × 10 hrs/day × 30 days × $1.12 = $420/month
```

**Realistic: $350-500/month**

---

## Scenario 5: Tiered (Cheap Always-On + Current для Burst)

### Setup

```
Endpoint 1 (cheap base):
  GPU: RTX 4090 @ $0.40/hr (найти дешевле!)
  min_workers: 1
  max_workers: 1
  Handles: 700 req/day
  Cost: $288/month

Endpoint 2 (current premium):
  GPU: RTX 4090 @ $1.12/hr
  min_workers: 0
  max_workers: 2
  Handles: 300 req/day (overflow)
  Active: ~2.5 hrs/day
  Cost: 2.5 × 30 × $1.12 = $84/month

TOTAL: $372/month
```

**Экономия vs pure always-on: $434/month (54%)**

---

## Comparison Table

| Strategy | Config | Cost/Month | Latency | Pros | Cons |
|----------|--------|------------|---------|------|------|
| **Pay-per-use** | min=0, idle=600 | $280-400 | 30-90s | Дешевле | Вариативная |
| **Always-on (current)** | min=1 | $806 | 20-30s | Стабильно | Очень дорого |
| **Always-on (cheap)** | min=1 @ $0.40/hr | $288 | 20-30s | Стабильно + дешево | Нужен другой GPU |
| **Smart scaling** | min=0, max=3 | $350-500 | 25-40s | Handles peaks | Средне |
| **Tiered** | 2 endpoints | $372 | 20-40s | Best balance | Сложнее setup |

---

## Recommendations

### Option A: Find Cheaper RTX 4090 (BEST)

```
RunPod Community Cloud: $0.30-0.40/hr
Vast.ai: $0.25-0.35/hr
Lambda Labs: $0.50/hr

Always-on cost @ $0.40/hr:
$288/month vs your $806/month

SAVE: $518/month!
```

**Action:**
1. Проверить RunPod Community Cloud
2. Проверить Vast.ai marketplace
3. Проверить Lambda Labs pricing

### Option B: Optimize Current Setup

Если нельзя найти дешевле:

```yaml
# Best config для 1000 req/day
min_workers: 0
max_workers: 2
idle_timeout: 900  # 15 минут (дольше между пиками)

# + Network Volume (легкий образ)
# + batch_size: 25
# + Все оптимизации
```

**Expected cost: $350-450/month**

### Option C: A100 может быть дешевле!

Проверьте цену A100:

```
A100 40GB стандартная цена: $1.10-1.40/hr

Если найдете A100 @ $1.30/hr:
- Быстрее генерация (20s вместо 30s)
- Можно больше batch_size
- Лучше для scaling

Pay-per-use cost:
1000 × 20s × 30 days / 3600 × $1.30 = $217/month
```

---

## Break-Even Analysis

### When is Always-On worth it?

```
Always-on: $806/month
Pay-per-use: $0.00031/sec

Break-even: $806 / $0.00031 = 2.6M seconds/month
= 722 hours/month
= 24 hours/day (всегда работает)

При 30s/request:
2.6M / 30 = 86,666 requests/month
= 2,888 requests/day

ВЫВОД: Always-on выгоден только при > 2,900 req/day
```

### Your case: 1000 req/day

```
Pay-per-use теоретически: $280/month (чистое время)
Pay-per-use реально: $350-450/month (с idle time)
Always-on: $806/month

SAVE with pay-per-use: $350-450/month (43-56%)
```

---

## Cost Per Request Analysis

| Strategy | Monthly Cost | Cost per Request | Notes |
|----------|-------------|------------------|-------|
| Pay-per-use (optimized) | $350 | $0.012/req | Best value |
| Always-on (current) | $806 | $0.027/req | Переплата 2.3x |
| Always-on (cheap @ $0.40) | $288 | $0.010/req | Best если найти |
| A100 pay-per-use | $217 | $0.007/req | Если быстрее gen |

---

## Action Plan

### Step 1: Check GPU pricing (priority!)

```bash
# RunPod
- Community Cloud RTX 4090
- Secure Cloud RTX 4090 в других регионах
- A100 40GB pricing

# Alternatives
- Vast.ai: https://vast.ai
- Lambda Labs: https://lambdalabs.com
```

### Step 2: Если цена не меняется

Используйте pay-per-use оптимально:

```yaml
min_workers: 0
max_workers: 2
idle_timeout: 900  # 15 min

# Network Volume setup (критично!)
# Легкий образ 2-3 GB
# batch_size: 25
```

**Expected: $350-450/month**

### Step 3: Monitor и optimize

```python
# Track metrics:
- Active hours per day
- Requests per hour distribution
- Cold starts per day
- Average latency

# Adjust idle_timeout based on patterns:
- Burst traffic: 300-600 sec
- Steady traffic: 600-900 sec
```

---

## GPU Marketplace Comparison

### RunPod

```
Secure Cloud (On-Demand):
  RTX 4090: $0.39-0.44/hr
  A100 40GB: $1.10-1.30/hr
  
Community Cloud (Spot):
  RTX 4090: $0.30-0.35/hr (может прерваться)
  A100 40GB: $0.80-1.00/hr
```

### Vast.ai

```
RTX 4090: $0.25-0.40/hr
A100 40GB: $0.70-1.20/hr

Pros: Дешевле
Cons: Меньше stability, нужен manual setup
```

### Lambda Labs

```
RTX 4090: $0.50/hr
A100 40GB: $1.10/hr

Pros: Простой setup
Cons: Часто нет availability
```

---

## Final Recommendation для 1000 req/day

### Best case: Найти RTX 4090 @ $0.40/hr

```
Always-on: $288/month
Стабильно, быстро, предсказуемо
```

### Realistic case: Текущая цена $1.12/hr

```
Pay-per-use с оптимизацией:
- min=0, idle=600
- Network Volume
- batch_size=25
- Легкий образ

Cost: $350-450/month
Latency: 30-50s (acceptable)
```

### Premium case: Нужна максимальная скорость

```
A100 40GB pay-per-use:
- Быстрее генерация (20s)
- Cost: $220-280/month
- Latency: 20-30s
```

**Bottom line: При текущей цене $1.12/hr используйте pay-per-use!**

