# RunPod Load Balancing Endpoints Guide

## Overview

Load Balancing endpoints in RunPod Serverless provide a new paradigm for deploying scalable video generation services. Unlike traditional queue-based endpoints, Load Balancing endpoints route requests directly to available workers, enabling better scalability and lower latency.

**Reference:** [RunPod Load Balancing Documentation](https://docs.runpod.io/serverless/load-balancing/overview)

## Key Concepts

### Queue-Based vs Load Balancing

| Aspect | Queue-Based (Current) | Load Balancing (New) |
|--------|----------------------|---------------------|
| Request Flow | Through queueing system | Direct to worker HTTP server |
| Implementation | Handler function | Custom HTTP server (FastAPI/Flask) |
| Access Pattern | `/run`, `/runsync` endpoints | Custom REST API paths |
| Latency | Higher (queue + worker) | Lower (single-hop) |
| Backpressure | Queue buffering | Request drop when overloaded |
| Error Recovery | Automatic retries | No built-in retry mechanism |
| Max Workers | Limited by GPU conflicts | Can scale to 100+ workers |

### When to Use Load Balancing

✅ **Use Load Balancing when:**
- You need to scale to 50+ workers
- Low latency is critical (real-time applications)
- You want custom REST API endpoints
- You need direct HTTP access to workers
- You can handle retries on the client side

❌ **Use Queue-Based when:**
- You need guaranteed request processing
- Automatic retries are required
- Simple handler pattern is sufficient
- Request processing takes > 5.5 minutes

## Architecture

### Current Architecture (Queue-Based)

```
Client Request
    ↓
RunPod Queue
    ↓
Worker 1 (handler function)
    ↓
Response
```

**Limitations:**
- Sequential processing per worker
- Queue overhead adds latency
- Difficult to scale beyond 10-20 workers
- Fixed `/run` and `/runsync` endpoints

### Load Balancing Architecture

```
Client Request
    ↓
https://ENDPOINT_ID.api.runpod.ai/generate
    ↓
RunPod Load Balancer
    ↓
    ├── Worker 1 (HTTP server) ← Request routed here
    ├── Worker 2 (HTTP server)
    ├── Worker 3 (initializing) ← Skipped (health check)
    └── ... Worker 100
```

**Advantages:**
- Direct routing to available workers
- Automatic health-based routing
- Custom REST API endpoints
- Can scale to 100+ workers
- Lower latency (no queue)

## Implementation

### Step 1: Convert Handler to HTTP Server

**Current Code (Queue-Based):**

```python
# runpod_server.py
import runpod

def handler(event):
    input_data = event.get("input", {})
    # ... processing ...
    return {"output": result}

runpod.serverless.start({"handler": handler})
```

**New Code (Load Balancing):**

```python
# runpod_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

class GenerationRequest(BaseModel):
    image: str  # base64 encoded
    audio: str  # base64 encoded
    cfg_scale: float = 1.0
    emotion: int = 8
    smooth: bool = False

@app.get("/ping")
async def health_check():
    """
    Required health check endpoint for Load Balancing.
    
    Returns:
        - 200: Worker is healthy and ready
        - 204: Worker is initializing (cold start)
        - Other: Worker is unhealthy
    """
    global pipe
    
    if pipe is None:
        # Worker is still initializing
        return {"status": "initializing"}, 204
    
    # Optional: Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            return {"status": "healthy", "gpu_available": True}, 200
        else:
            return {"status": "unhealthy", "error": "GPU not available"}, 503
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503

@app.post("/generate")
async def generate(request: GenerationRequest):
    """
    Generate video from image and audio.
    
    This endpoint replaces the queue-based handler function.
    """
    # Acquire GPU lock (still needed for thread safety)
    gpu_lock.acquire()
    try:
        # Your existing processing logic from handler()
        result = process_generation(
            image=request.image,
            audio=request.audio,
            cfg_scale=request.cfg_scale,
            emotion=request.emotion,
            smooth=request.smooth
        )
        return {"output": result}
    finally:
        gpu_lock.release()

@app.get("/cuda")
async def check_cuda():
    """CUDA diagnostics endpoint"""
    cuda_info = check_cuda_info()
    return {"cuda_info": cuda_info}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Step 2: Update Dependencies

Add to `requirements.txt`:

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6  # For file uploads if needed
```

### Step 3: Environment Variables

**Required Environment Variables:**

```bash
PORT=80                    # Main application port
PORT_HEALTH=80             # Health check port (usually same as PORT)
USE_EAGER_INIT=false       # For cost optimization
```

**In RunPod Dashboard:**
1. Go to Endpoint Settings
2. Add Environment Variables:
   - `PORT`: `80`
   - `PORT_HEALTH`: `80`
   - `USE_EAGER_INIT`: `false` (or `true` for always-on workers)

### Step 4: Container Configuration

**In RunPod Dashboard → Endpoint Settings:**

1. **Endpoint Type:** Select "Load Balancing" (not "Queue-based")
2. **Expose HTTP Ports:** Add port `80`
3. **Container Image:** `krapiunitski12/moda-runpod:v1.0.0`
4. **GPU:** RTX 4090 or A6000/A40
5. **Min Workers:** `0` (auto-scale) or `1` (always-on)
6. **Max Workers:** `100` (or your desired limit)

## Configuration

### Recommended Settings for 100 Workers

```yaml
Endpoint Type: Load Balancing
Container Image: krapiunitski12/moda-runpod:v1.0.0
GPU: RTX 4090 @ $0.40/hr (cheaper region) or A6000/A40 @ $1.22/hr
Min Workers: 0          # Auto-scale (cost-effective)
Max Workers: 100         # Maximum concurrent workers
Idle Timeout: 600        # 10 minutes

Environment Variables:
  PORT: 80
  PORT_HEALTH: 80
  USE_EAGER_INIT: false  # For pay-per-use
  HUGGINGFACE_USERNAME: your-username
```

### Cost Optimization

**Pay-Per-Use (min_workers=0):**
```
10,000 req/day × 30 sec/req = 83.3 hrs/day
83.3 hrs/day × 30 days × $0.40/hr = $1,000/мес
+ idle overhead = ~$1,200-1,500/мес
```

**Always-On (min_workers=100):**
```
100 workers × 24 hrs × 30 days × $0.40/hr = $28,800/мес
```

## Request Flow

### Client Implementation

```python
import requests
import time
import base64

class MoDAClient:
    def __init__(self, endpoint_id: str, api_key: str):
        self.base_url = f"https://{endpoint_id}.api.runpod.ai"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def health_check_with_retry(self, max_retries=3, delay=5):
        """Wait for worker to be ready (handles cold start)"""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/ping",
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code == 200:
                    print("✓ Worker is healthy")
                    return True
                elif response.status_code == 204:
                    print(f"⏳ Worker initializing (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                else:
                    print(f"✗ Worker unhealthy: {response.status_code}")
                    return False
            except Exception as e:
                print(f"✗ Health check failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
        
        return False
    
    def generate(self, image_path: str, audio_path: str, **kwargs):
        """Generate video from image and audio"""
        # Ensure worker is ready
        if not self.health_check_with_retry():
            raise RuntimeError("Worker not available")
        
        # Read and encode files
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        with open(audio_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        
        # Send request
        payload = {
            "image": image_b64,
            "audio": audio_b64,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=payload,
            timeout=330  # 5.5 minutes max
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            raise RuntimeError("No workers available (timeout)")
        elif response.status_code == 524:
            raise RuntimeError("Processing timeout (>5.5 minutes)")
        else:
            raise RuntimeError(f"Request failed: {response.status_code}")

# Usage
client = MoDAClient("your-endpoint-id", "your-api-key")
result = client.generate("image.jpg", "audio.mp3", emotion=8)
```

## Timeouts and Limits

### Request Timeouts

1. **Request Timeout (2 minutes):**
   - If no worker is available within 2 minutes
   - Returns `400` error
   - **Solution:** Implement retries with exponential backoff

2. **Processing Timeout (5.5 minutes):**
   - Maximum time for request processing
   - Returns `524` error if exceeded
   - **Solution:** Optimize processing or use queue-based for long tasks

### Payload Limits

- **Request/Response:** 30 MB maximum
- **Solution for large files:**
  - Use Network Volume for model artifacts
  - Implement chunking strategies
  - Store large files in S3-compatible storage

## Health Check Implementation

### Required Endpoint: `/ping`

```python
@app.get("/ping")
async def health_check():
    """
    Health check endpoint required by RunPod Load Balancer.
    
    Status Codes:
        - 200: Worker is healthy and ready to process requests
        - 204: Worker is initializing (cold start)
        - 503: Worker is unhealthy (will be removed from pool)
    """
    global pipe
    
    # Check if pipeline is initialized
    if pipe is None:
        return {"status": "initializing"}, 204
    
    # Optional: Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "unhealthy", "error": "GPU not available"}, 503
        
        # Check GPU memory
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        if mem_allocated > 45:  # 45GB out of 48GB (A6000/A40)
            return {"status": "unhealthy", "error": "GPU memory full"}, 503
        
        return {
            "status": "healthy",
            "gpu_available": True,
            "memory_allocated_gb": round(mem_allocated, 2)
        }, 200
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503
```

### Health Check Behavior

- **Cold Start:** `/ping` returns `204` until pipeline is initialized
- **Ready:** `/ping` returns `200` when worker is ready
- **Unhealthy:** Any other status code removes worker from pool
- **Frequency:** RunPod checks health periodically

## Thread Safety (Still Required!)

Even with Load Balancing, thread safety is critical:

```python
# Global locks (still needed)
gpu_lock = threading.Lock()
pipe_init_lock = threading.Lock()

@app.post("/generate")
async def generate(request: GenerationRequest):
    # Acquire lock before GPU operations
    gpu_lock.acquire()
    try:
        pipe = get_pipe()  # Thread-safe initialization
        # ... GPU operations ...
    finally:
        gpu_lock.release()
```

**Why locks are still needed:**
- Multiple requests can arrive at the same worker
- FastAPI handles requests in separate threads
- CUDA operations are not thread-safe
- Pipeline initialization must be protected

## Migration Guide

### From Queue-Based to Load Balancing

1. **Install FastAPI:**
   ```bash
   pip install fastapi uvicorn[standard]
   ```

2. **Convert handler to FastAPI endpoints:**
   - Replace `handler(event)` with FastAPI route handlers
   - Extract input from request body instead of `event["input"]`
   - Return JSON responses directly

3. **Add health check endpoint:**
   - Implement `/ping` endpoint
   - Return `204` during initialization
   - Return `200` when ready

4. **Update RunPod configuration:**
   - Change endpoint type to "Load Balancing"
   - Set `PORT` and `PORT_HEALTH` environment variables
   - Expose port 80 in container configuration

5. **Update client code:**
   - Use REST API endpoints instead of `/run` or `/runsync`
   - Implement health check with retries
   - Handle `400` and `524` error codes

## Best Practices

### 1. Health Check Retries

Always implement retries for cold starts:

```python
def health_check_with_retry(base_url, api_key, max_retries=5, delay=5):
    """Wait for worker to be ready"""
    for attempt in range(max_retries):
        response = requests.get(f"{base_url}/ping", headers=headers)
        if response.status_code == 200:
            return True
        time.sleep(delay)
    return False
```

### 2. Error Handling

Handle specific error codes:

```python
if response.status_code == 400:
    # No workers available - retry with backoff
elif response.status_code == 524:
    # Processing timeout - request too long
elif response.status_code == 502:
    # Worker misconfigured - check ports
```

### 3. Monitoring

Monitor worker health and performance:

```python
@app.get("/metrics")
async def metrics():
    """Optional metrics endpoint for monitoring"""
    return {
        "active_workers": get_active_worker_count(),
        "queue_length": get_queue_length(),
        "avg_processing_time": get_avg_processing_time()
    }
```

### 4. Cost Optimization

- Use `min_workers=0` for pay-per-use
- Set appropriate `idle_timeout` (600 seconds = 10 minutes)
- Use cheaper GPU regions when possible
- Monitor actual usage vs costs

## Troubleshooting

### Issue: "No workers available" (400 error)

**Cause:** All workers are busy or initializing

**Solutions:**
- Increase `max_workers` in endpoint settings
- Implement retries with exponential backoff
- Check worker health via `/ping` endpoint
- Reduce `idle_timeout` to keep more workers warm

### Issue: Processing timeout (524 error)

**Cause:** Request processing exceeds 5.5 minutes

**Solutions:**
- Optimize processing (reduce batch size, shorter videos)
- Use queue-based endpoints for long tasks
- Split processing into multiple requests

### Issue: Worker stays unhealthy

**Cause:** Health check returns non-200/204 status

**Solutions:**
- Check `/ping` endpoint implementation
- Verify GPU availability
- Check port configuration (`PORT`, `PORT_HEALTH`)
- Review worker logs for errors

### Issue: Port not exposed (502 error)

**Cause:** Port not configured in container settings

**Solutions:**
- Add port 80 to "Expose HTTP Ports" in endpoint settings
- Verify `PORT` environment variable matches exposed port
- Check container logs for port binding errors

## Performance Comparison

### Queue-Based (Current)

```
Request → Queue → Worker → Response
Latency: ~50-100ms (queue overhead)
Throughput: Limited by queue processing
Max Workers: ~10-20 (GPU conflicts)
```

### Load Balancing (New)

```
Request → Load Balancer → Worker → Response
Latency: ~10-20ms (direct routing)
Throughput: Scales with worker count
Max Workers: 100+ (automatic distribution)
```

## Cost Analysis

### Scenario: 10,000 requests/day

**Load Balancing (Pay-Per-Use):**
```
10,000 req/day × 30 sec/req = 83.3 hrs/day
83.3 hrs/day × 30 days × $0.40/hr = $1,000/мес
+ idle overhead = ~$1,200-1,500/мес
```

**Load Balancing (Always-On, 100 workers):**
```
100 workers × 24 hrs × 30 days × $0.40/hr = $28,800/мес
```

**Queue-Based (100 separate endpoints):**
```
100 endpoints × $806/мес = $80,600/мес (always-on)
or
100 endpoints × $1,200/мес = $120,000/мес (pay-per-use)
```

**Savings with Load Balancing:** 60-75% cost reduction

## References

- [RunPod Load Balancing Overview](https://docs.runpod.io/serverless/load-balancing/overview)
- [Build a Load Balancing Worker](https://docs.runpod.io/serverless/load-balancing/build-a-load-balancing-worker)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CUDA Device Busy Fix](./cuda-device-busy-fix.md)

## Summary

Load Balancing endpoints provide:
- ✅ **Scalability:** Scale to 100+ workers with one endpoint
- ✅ **Performance:** Lower latency (direct routing)
- ✅ **Flexibility:** Custom REST API endpoints
- ✅ **Cost Efficiency:** 60-75% cost savings vs multiple endpoints
- ✅ **Simplicity:** Automatic load balancing and health checks

**Next Steps:**
1. Convert handler to FastAPI HTTP server
2. Add `/ping` health check endpoint
3. Update RunPod endpoint configuration
4. Update client code for REST API
5. Test with small `max_workers` first, then scale up

