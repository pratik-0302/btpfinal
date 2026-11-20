"""Start the local application:  python server.py   ->  http://127.0.0.1:8000

Binds to 127.0.0.1 only (other machines cannot connect) and runs a single API process,
which starts a single inference worker that loads the model once.
"""
import uvicorn

import config

if __name__ == "__main__":
    print(f"Lip-sync deepfake detector: http://{config.HOST}:{config.PORT}  (local only; Ctrl+C to stop)")
    uvicorn.run("api.app:app", host=config.HOST, port=config.PORT, workers=1, log_level="info")
