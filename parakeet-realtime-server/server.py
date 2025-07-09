import base64
import logging
from contextlib import asynccontextmanager

import logfire
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
    logfire.info('Loading model ...')
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    logfire.info('Server ready')
    yield


app = FastAPI(lifespan=lifespan)

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_fastapi(app)


@app.post('/transcribe')
async def transcribe(audio_file: UploadFile = File(...)):
    await audio_file.read()

    return


@app.websocket('/transcribe')
async def transcribe_ws(websocket: WebSocket):
    global model
    await websocket.accept()

    logfire.info('Client connected')

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
                if len(segment.words) > 0:
                    await websocket.send_json(segment.model_dump())

    except starlette.websockets.WebSocketDisconnect:
        logfire.info('Client disconnected')

    except Exception as e:
        logfire.error(f'Exception {e}', _exc_info=True)


@app.get('/health')
async def health():
    return {'status': 'healthy'}
