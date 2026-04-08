import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=[],
    allow_headers=[],
)

# Carica modello ONNX all'avvio
session = None

@app.on_event(startup)
async def startup()
    global session
    session = ort.InferenceSession(captcha_model.onnx)
    print(✅ Modello ONNX caricato)

def preprocess(image_bytes)
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((150, 49))          # (width, height)
    img_array = np.array(img, dtype=np.float32)  255.0
    # Shape (1, 1, 49, 150) perché ONNX si aspetta (batch, channel, height, width)
    img_tensor = img_array[np.newaxis, np.newaxis, ...]
    return img_tensor

@app.post(solve)
async def solve(file UploadFile = File(...))
    if session is None
        return JSONResponse(status_code=503, content={error Modello non caricato})
    try
        image_bytes = await file.read()
        img_tensor = preprocess(image_bytes)
        # L'output del modello ONNX è una lista di 4 array (uno per cifra)
        outputs = session.run(None, {input img_tensor})
        captcha = ''.join(str(np.argmax(out)) for out in outputs)
        return JSONResponse(content={success True, captcha captcha})
    except Exception as e
        return JSONResponse(status_code=500, content={error str(e)})

@app.get(health)
async def health()
    return {status ok}

if __name__ == __main__
    import uvicorn
    uvicorn.run(app, host=0.0.0.0, port=8000)