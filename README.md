# Assignment 4 

This builds on the Module 6 class activity (a `helper_lib` package with a GAN) by adding two CIFAR-10 image generators and serving all three through FastAPI in Docker:


## Run with Docker

```bash
docker build -t assignment4-gan-api .
```

```bash
docker run --rm -p 8000:8000 assignment4-gan-api
```

Then open http://localhost:8000/docs for the interactive Swagger UI, or:

```bash
curl "http://localhost:8000/generate/diffusion?num_samples=16&steps=50" -o diffusion.png
```

```bash
curl "http://localhost:8000/generate/ebm?num_samples=16&steps=256" -o ebm.png
```

```bash
curl "http://localhost:8000/generate/gan?num_samples=16" -o gan.png
```
