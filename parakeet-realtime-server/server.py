import base64
import logging
from contextlib import asynccontextmanager

import numpy as np
import starlette
from fastapi import FastAPI, UploadFile, File
from starlette.websockets import WebSocket

logging.disable(logging.CRITICAL)

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecRNNTBPEModel

from transcriber import continuous_transcriber

model: EncDecRNNTBPEModel | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model

    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    yield


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)


@app.post('/transcribe')
async def transcribe(audio_file: UploadFile = File(...)):
    await audio_file.read()

    return


@app.websocket('/transcribe')
async def transcribe_ws(websocket: WebSocket):
    global model
    await websocket.accept()

    async def samples_generator():
        while True:
            data = await websocket.receive_json()

            if data.get('commit', False):
                yield None

            else:
                samples = base64.b64decode(data['samples'])
                samples = np.frombuffer(samples, dtype=np.float32)

                yield samples

    try:
        while True:
            async for segment in continuous_transcriber(model, samples_generator()):
                if len(segment['words']) > 0:
                    await websocket.send_json(segment)

    except starlette.websockets.WebSocketDisconnect:
        pass

    except Exception as e:
        logger.error('Exception occurred', exc_info=True)
        logger.error(str(e))


@app.get('/health')
async def health():
    return {'status': 'healthy'}
