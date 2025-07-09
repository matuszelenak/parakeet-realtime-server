import argparse
import asyncio
import base64
import json
import os
import random

import scipy.io.wavfile as wav
import websockets
from websockets.asyncio.client import ClientConnection

MIN_CHUNK_SAMPLES = 500
MAX_CHUNK_SAMPLES = 2000

async def send_wav_data(websocket: ClientConnection, path: str):
    try:
        sample_rate, audio_data = wav.read(path)
        total_samples = audio_data.shape[0]

        frames_sent = 0
        while frames_sent < total_samples:
            chunk_size = random.randint(MIN_CHUNK_SAMPLES, MAX_CHUNK_SAMPLES)
            remaining_frames = total_samples - frames_sent
            frames_to_read = min(chunk_size, remaining_frames)

            sample_batch = audio_data[frames_sent:frames_sent + frames_to_read]

            encoded_data = base64.b64encode(sample_batch.tobytes()).decode('utf-8')
            payload = {
                "samples": encoded_data
            }

            await websocket.send(json.dumps(payload))

            frames_sent += frames_to_read

            await asyncio.sleep(frames_to_read / sample_rate / 2)

        await websocket.send(json.dumps({"commit": True}))

    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        await websocket.send(json.dumps({"error": "file_not_found"}))
    except Exception as e:
        print(f"An error occurred in send_wav_data: {e}")
    finally:
        await websocket.close()


async def receive_messages(websocket):
    try:
        async for message in websocket:
            j = json.loads(message)
            if j['complete']:
                print(' '.join(j['words']), end=' ', flush=True)

    except websockets.exceptions.ConnectionClosed:
        print("Receiver: Connection closed.")
    except Exception as e:
        print(f"An error occurred in receive_messages: {e}")


async def run_client(uri, file_path):
    """
    Connects to the websocket server and runs sender/receiver tasks.
    """
    try:
        async with websockets.connect(uri, ping_timeout=None, open_timeout=None, close_timeout=None) as websocket:
            print(f"Successfully connected to WebSocket server at {uri}")

            sender_task = asyncio.create_task(send_wav_data(websocket, file_path))
            receiver_task = asyncio.create_task(receive_messages(websocket))

            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.ALL_COMPLETED,
            )

            for task in pending:
                task.cancel()

    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running at {uri}?")
    except Exception as e:
        print(f"An error occurred while connecting or running client: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_file", help="The path to the WAV file you want to stream.")
    parser.add_argument("--uri", default="ws://localhost:8765", help="The WebSocket server URI to connect to.")

    args = parser.parse_args()

    if not os.path.exists(args.wav_file):
        print(f"Error: The specified WAV file was not found at '{args.wav_file}'")
    else:
        try:
            asyncio.run(run_client(args.uri, args.wav_file))
        except KeyboardInterrupt:
            print("\nClient manually shut down.")
