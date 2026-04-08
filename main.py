import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = None

@app.on_event("startup")
async def startup():
    global session
    # Specifica i providers esplicitamente (solo CPU)
    session = ort.InferenceSession("captcha_model.onnx", providers=['CPUExecutionProvider'])
    print("[OK] Modello ONNX caricato")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((150, 49))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = img_array[np.newaxis, np.newaxis, ...]  # (1,1,49,150)
    return img_tensor

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    if session is None:
        return JSONResponse(status_code=503, content={"error": "Modello non caricato"})
    try:
        image_bytes = await file.read()
        img_tensor = preprocess(image_bytes)
        outputs = session.run(None, {"input": img_tensor})
        captcha = ''.join(str(np.argmax(out)) for out in outputs)
        return JSONResponse(content={"success": True, "captcha": captcha})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
