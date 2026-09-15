# Vercel browser demo

This is a static client for the FastAPI service in `api/main.py`. Deploy the
`web/` directory as a Vercel project, then open the resulting URL with your
API URL as a query parameter:

```text
https://your-demo.vercel.app/?api=https://your-api.example.com
```

The model itself should run on a long-lived CPU/GPU host (Docker, a VM, or a
Hugging Face Space). Vercel serves this lightweight browser UI and is not a
good host for a 134 MB PyTorch checkpoint.
