# TIGAS

Text to Image Generation API Server

## API

### /api/v1

#### /generate

POST request to generate an image from a text.

API for the text inversion model.

### /api/v2

#### /generate

POST request to generate an image from a text prompt and a image prompt (image to image with text guidance).

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

3. Clean up expired node asset results

    - Clean up expired node asset results
    - Cron job


4. Parse logs

    - Parse logs as a dataframe
    - Cron job

5. Reload Applications

    - Reload core applications
    - Move logs to the backup directory
    - Cron job
