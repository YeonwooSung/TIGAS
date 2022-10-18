# Test TIGAS

## Model

First, you could test the model with the [provided jupyter notebook](./example_codes/stable_diffusion_custom_model.py).

Also, you could test the model inference by running the following command:

```bash
python3 tigas/api/utils/model.py
```

This will generate a sample image from a prompt "Dogs running on a beach".

## API

### API v1

The v1 API supports the text inversion.

The /api/v1/generate/tti endpoint looks for the JSON object in the request body, where the JSON contains the prompt text.

Below is an example of a request to the /api/v1/generate/tti endpoint:

```bash
curl -d '{"text":"a beautiful cinematic sexy female sea goddes, water wings , fantasy sea landscape, fantasy magic, undercut hairstyle, short aqua blue black fade hair, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Greg Rutkowski and Alphonse Mucha, masterpiece"}' -H "Content-Type: application/json" -X POST "http://34.64.108.168:5000/api/v1/generate/tti"
```

### API v2

The v2 API supports the image2image diffusion, which generates image from guidance prompt and sample image.

The /api/v2/generate/i2i endpoint looks for the multipart/form-data with the following fields:

* text: the guidance prompt
* image: the sample image file (uploaded as a file)

example curl command:

```bash
curl -F "text=4k digital illustration" -F "image=@/Users/YeonwooSung/Desktop/input.jpeg" -X POST http://localhost:5000/api/v2/generate/i2i
```
