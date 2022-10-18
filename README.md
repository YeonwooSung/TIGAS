# TIGAS

Text to Image Generation API Server

## Deploy with PM2

```bash
pm2 start tigas/app.py --interpreter venv/bin/python

pm2 start cron/delete_old_files.py --interpreter venv/bin/python

pm2 start cron/cleanup_node_asset_results.py --interpreter venv/bin/python

pm2 start cron/parse_logs.py --interpreter venv/bin/python

pm2 start cron/reloadCiCd.py --interpreter venv/bin/python
```

## Processes

1. TIGAS API server

    - Text-to-image generation API server
    - FastAPI based API server

2. Clean up expired images

    - Clean up expired images
    - Cron job
