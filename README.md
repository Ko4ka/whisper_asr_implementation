# whisper_asr_implementation

# CTRL+C / CTRL + V

```
CD C:\Users\Alex\whisper_asr_implementation

venv\Scripts\activate

uvicorn service_asr_bulk_new:app --host 127.0.0.1 --port 8000

uvicorn service_diar_bulk_new:app --host 127.0.0.1 --port 8001

uvicorn service_allignment:app --host 127.0.0.1 --port 8002

uvicorn frontman:app --host 127.0.0.1 --port 8005

```
