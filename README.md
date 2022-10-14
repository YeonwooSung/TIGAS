# TIGAS

Text to Image Generation API Server

## Deploy with PM2

```bash
pm2 start tigas/app.py --interpreter venv/bin/python
```

## Processes

1. TIGAS API server

    - Text-to-image generation API server
    - FastAPI based API server

2. Clean up expired images

    - Clean up expired images
    - Cron job
