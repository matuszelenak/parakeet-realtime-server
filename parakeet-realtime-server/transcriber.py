import logging
from typing import List, TypeVar, AsyncGenerator, TypedDict

import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils import Hypothesis

T = TypeVar("T")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.ERROR)

SAMPLING_RATE = 16000


def split_to_batches(array: List[T], batch_size: int) -> List[List[T]]:
    return [array[i:i + batch_size] for i in range(0, len(array), batch_size)]


async def even_chunks(audio_chunks: AsyncGenerator[np.ndarray, None], chunk_size):
    buffer: np.ndarray = np.zeros((chunk_size, ), np.float32)
    i = 0
    async for chunk in audio_chunks:
        if chunk is None:
            break

        for sample in chunk:
            buffer[i] = sample

            i += 1
            if i == chunk_size:
                yield buffer

                i = 0
                buffer = np.zeros((chunk_size, ), np.float32)

    if i > 0:
        yield buffer[:i]

    yield None


class Word(TypedDict):
    start: float
    end: float
    word: str


async def continuous_transcriber(
        model: EncDecRNNTBPEModel,
        audio_chunks: AsyncGenerator[np.ndarray, None],
        stable_words_for_iterations=4,
        stable_words_before_advance=3
) -> AsyncGenerator[dict, None]:
    transcribed_samples = np.array([], dtype=np.float32)

    segments: List[List[Word]] = []

    stable_segment_counter = 0
    current_ts_offset = 0.0
    transcribed_chain: List[Word] = []
    sample_offset = 0

    last_confirmed_word = ''
    async for chunk in even_chunks(audio_chunks, 4000):
        if chunk is None:
            yield {
                'complete': True,
                'words': [w['word'] for w in transcribed_chain],
                'start': current_ts_offset,
                'id': stable_segment_counter,
                'samples': transcribed_samples.shape[0],
                'final': True
            }

            return

        transcribed_samples = np.concatenate((transcribed_samples, chunk), axis=0)
        if transcribed_samples.shape[0] < (SAMPLING_RATE / 2):
            continue

        logger.info(f'Processing audio with length {transcribed_samples.shape[0] / SAMPLING_RATE}s')
        output: List[Hypothesis] = model.transcribe(transcribed_samples, timestamps=True, verbose=False)

        transcribed_chain: List[Word] = output[0].timestamp['word']

        if len(transcribed_chain) == 0:
            continue

        if transcribed_chain[0]['word'].strip().lower() == last_confirmed_word.lower():
            transcribed_chain = transcribed_chain[1:]

        segments.append(transcribed_chain)

        best_segment_index, best_segment_length = -1, -1
        for ref_s_i, ref_segment_words in enumerate(segments):
            if len(ref_segment_words) < stable_words_before_advance:
                continue

            ref_sequence = ' '.join([w['word'].lower().strip() for w in ref_segment_words])
            sequence_repetitions = 1

            for com_segment_words in segments[ref_s_i + 1:]:
                comp_sequence = ' '.join([w['word'].lower().strip() for w in com_segment_words])

                if comp_sequence[:len(ref_sequence)] == ref_sequence:
                    sequence_repetitions += 1

            if sequence_repetitions >= stable_words_for_iterations:
                if len(ref_segment_words) > best_segment_length:
                    best_segment_length = len(ref_segment_words)
                    best_segment_index = ref_s_i

        if best_segment_index != -1:
            best_s_words = segments[best_segment_index]

            yield {
                'complete': True,
                'words': [w['word'] for w in best_s_words[:-1]],
                'start': current_ts_offset,
                'id': stable_segment_counter,
                'samples': transcribed_samples.shape[0]
            }

            last_confirmed_word = best_s_words[:-1][-1]['word'].strip().lower()

            samples_to_advance = int(best_s_words[-2]['end'] * SAMPLING_RATE)
            transcribed_samples = transcribed_samples[samples_to_advance:]
            segments = segments[best_segment_index + 1:]
            stable_segment_counter += 1
            current_ts_offset += samples_to_advance / SAMPLING_RATE
            sample_offset += samples_to_advance

        else:
            yield {
                'complete': False,
                'words': [w['word'] for w in transcribed_chain],
                'samples': transcribed_samples.shape[0],
                'start': current_ts_offset,
            }
