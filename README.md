# Parakeet realtime transcription server

A simple server that provides low-latency streamed audio transcription.
Mostly a result of frustration with the other available OSS solutions.
It is not meant to be high-performance or serve multiple users concurrently and was developed mainly as a subcomponent of my [voice assistant project](https://github.com/matuszelenak/Her.git)

## Algorithm

In an endless loop we perform the following:
- The server receives chunks of audio samples over a websocket connection (the chunk can have an arbitrary length).
- It accumulates these chunks to a buffer until a minimum viable duration is achieved and then attempts to transcribe this buffer using the Parakeet STT model.
- The result of such transcription is a chain of words with timestamps relative to this buffer.
- We continually append new received samples to the aforementioned buffer and transcribe it each time, gathering multiple chains of transcribed words.
- After each such transcription round, we traverse the collection of word chains from the most to least recent, choosing one as a reference and comparing it to the ones preceding it, taking note of how long a word prefix for how many iterations we have.
- If the word prefix of a specified minimum length has been "stable" for a specified number of past iterations, we commit this transcription as final. We then move up in the accumulated sample buffer, so that the next transcriptions are no longer performed on the samples that include the commited words.
- We send the finalized transcription over the websocket back

## Running the server

For now, only systems with a CUDA-capable GPU are supported.

Run `docker-compose up --build --watch` in the root of the project.

## Usage

The server has a websocket endpoint `/transcribe` that accepts only two possible JSON messages:
```json
{
  "samples": "base64encodedbinarysampledata"
}
```

Where the base64 encoded data must be a 16Khz 32-bit float PCM audio samples.
The other possible message is:

```json
{
  "commit": "true"
}
```
that signals to the server that the currently accumulated audio samples should be evaluated and returned as finished.

## Example script

With the server running you can run the `example.py` script on a compatible audio file (16KHz 32-bit float sample WAV file) and receive live transcription.

```shell
python3 example.py --uri ws://localhost:9090/transcribe /path/to/file.wav
```

