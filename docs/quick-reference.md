# Quick Reference - RunPod Optimization

## Commands Cheatsheet

### Network Volume Setup (one-time)

```bash
# В RunPod temporary pod с SSH:
pip install huggingface-hub

export HF_TOKEN="hf_xxx"
export HF_USERNAME="your-username"

python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_USERNAME}/moda-pretrain-weights',
    local_dir='/workspace/pretrain_weights',
    local_dir_use_symlinks=False,
    token='${HF_TOKEN}'
)
"

# Проверка:
ls -la /workspace/pretrain_weights/moda/
ls -la /workspace/pretrain_weights/decode/
```

### Docker Build (легкий образ)

```bash
# Экспорт переменных
export HUGGINGFACE_USERNAME="your-username"
export HF_TOKEN="hf_xxx"
export DOCKER_USER="your-docker-username"

# Build БЕЗ моделей (после удаления секции из Dockerfile)
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t ${DOCKER_USER}/moda-runpod:lightweight \
  -f Dockerfile \
  .

# Проверка размера (должен быть 2-3 GB)
docker images | grep moda-runpod

# Push
docker push ${DOCKER_USER}/moda-runpod:lightweight
```

### Проверка производительности

```bash
# Локальный тест размера образа
docker images moda-runpod:lightweight

# RunPod API test
curl -X POST https://api.runpod.ai/v2/YOUR-ENDPOINT/run \
  -H "Authorization: Bearer YOUR-API-KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "base64...",
      "audio": "base64..."
    }
  }'
```

---

## Configuration Files

### configs/audio2motion/inference/inference.yaml

```yaml
# Performance optimization
batch_size: 25           # RTX 4090: 20-25, A100: 50-75
max_video_length: 500    # 20 сек @ 25fps

# Video quality (fast encoding)
crf: 27                  # 25-28 (выше = быстрее, ниже качество)
codec: "libx264"
preset: "veryfast"       # veryfast | faster | fast
```

### RunPod Endpoint Settings

```yaml
# Start configuration
Container Image: username/moda-runpod:lightweight
Network Volume: /workspace → moda-models (20 GB)

Environment Variables:
  HUGGINGFACE_USERNAME: your-username
  HF_TOKEN: hf_xxx  # опционально после setup

GPU: RTX 4090 (24GB)
min_workers: 0
max_workers: 2
idle_timeout: 600  # 10 минут

# Production (при росте нагрузки)
min_workers: 1
max_workers: 3
```

---

## Performance Targets

### Cold Start (новый worker)
```
Легкий образ (2-3 GB):     75-100 сек
Тяжелый образ (14.62 GB):  240 сек
```

### Generation (hot worker)
```
batch_size=10:   40-50 сек
batch_size=25:   25-35 сек ← target
batch_size=50:   15-20 сек (A100)
```

---

## Cost Estimates (RTX 4090 @ $0.00031/sec = $1.12/hr)

```
Always-on (min=1):           $806/мес  ($1.12 × 24 × 30)
Half-day (12hrs):            $403/мес  ($1.12 × 12 × 30)
Pay-per-use (50 req/day):    
  - 30s/req: $16/мес   (50 × 30s × 30 × $1.12/3600)
  - 60s/req: $33/мес
  - 90s/req: $50/мес
```

---

## Troubleshooting Quick Fixes

### OOM Error
```yaml
# inference.yaml
batch_size: 15  # уменьшить
```

### Slow cold start
```bash
# Проверить размер образа
docker images

# Должен быть < 5 GB для быстрого старта
```

### Models not found
```python
# runpod_server.py - проверить пути
ls /workspace/pretrain_weights/
ls /app/pretrain_weights/
```

### Долгая генерация
```yaml
# inference.yaml - увеличить batch
batch_size: 25

# и/или короче видео
max_video_length: 250  # 10 сек
```

---

## Decision Matrix

### Когда использовать что?

| Нагрузка | Config | Cost/мес ($1.12/hr) | Latency |
|----------|--------|----------|---------|
| < 10 req/day | min=0, idle=300 | $30-80 | 60-90s |
| 10-50 req/day | min=0, idle=600 | $80-250 | 30-50s |
| 50-100 req/day | min=0, idle=600 | $250-500 | 25-40s |
| 100-500 req/day | min=0, idle=900 | $300-450 | 25-40s |
| 1000 req/day | min=0, idle=900 | $350-450 | 30-50s |
| > 2900 req/day | min=1 | $806 | 20-30s |

**Note:** При 1000 req/day НЕ используйте always-on! Pay-per-use экономит $350-450/мес

### Когда увеличивать batch_size?

```
RTX 4090 24GB:
  Safe:      batch_size: 20
  Optimal:   batch_size: 25
  Risky:     batch_size: 30+

A100 40GB:
  Safe:      batch_size: 50
  Optimal:   batch_size: 75
  Max:       batch_size: 100
```

---

## Monitoring Commands

```bash
# RunPod logs
runpod logs YOUR-ENDPOINT

# Timing breakdown в логах:
# [INFO] Validation: X.XXs
# [INFO] Decode: X.XXs
# [INFO] Audio conversion: X.XXs
# [INFO] Inference: X.XXs  ← главная метрика
# [INFO] Encode: X.XXs
# [INFO] Total: X.XXs

# GPU memory (в pod):
nvidia-smi

# Disk usage:
df -h /workspace
```

---

## Migration Checklist

- [ ] Network Volume создан
- [ ] Модели загружены в volume
- [ ] Dockerfile обновлен (удалена секция моделей)
- [ ] runpod_server.py обновлен (новая логика ensure_models)
- [ ] Образ пересобран и < 5 GB
- [ ] Образ загружен в Docker Hub
- [ ] Endpoint обновлен (новый образ + volume)
- [ ] batch_size увеличен до 25
- [ ] preset изменен на "veryfast"
- [ ] Тест cold start < 100 сек
- [ ] Тест hot generation < 35 сек
- [ ] Cost мониторинг настроен

---

## Emergency Rollback

Если что-то пошло не так:

```yaml
# Вернуться к старой конфигурации
Container Image: username/moda-runpod:latest  # старый образ 14.62 GB
Network Volume: отключить

# Старые настройки
batch_size: 10
preset: "faster"
```

Все будет работать как раньше (медленно, но стабильно).

