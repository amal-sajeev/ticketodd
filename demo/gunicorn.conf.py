import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
worker_class = "uvicorn.workers.UvicornWorker"
workers = int(os.getenv("WEB_CONCURRENCY", (2 * multiprocessing.cpu_count()) + 1))

# Long timeout for OpenAI API calls (can take 2-10s)
timeout = 120
graceful_timeout = 30
keepalive = 5

# Restart workers periodically to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload app so workers share memory for imports
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

# Worker tmp dir — use /dev/shm in containers for faster heartbeat
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None

forwarded_allow_ips = "*"
proxy_protocol = False
