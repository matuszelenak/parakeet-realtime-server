from typing import List, AsyncGenerator, TypedDict

import logfire
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils import Hypothesis

from models import TranscribedSegment
from settings import settings

SAMPLING_RATE = 16000


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
        audio_chunks: AsyncGenerator[np.ndarray, None]
) -> AsyncGenerator[TranscribedSegment, None]:
    transcribed_samples = np.array([], dtype=np.float32)

    segments: List[List[Word]] = []

    stable_segment_counter = 0
    current_ts_offset = 0.0
    transcribed_chain: List[Word] = []

    last_confirmed_word = ''
    async for chunk in even_chunks(audio_chunks, 4000):
        if chunk is None:
            yield TranscribedSegment(
                complete=True,
                words=[w['word'] for w in transcribed_chain],
                start=current_ts_offset,
                id=stable_segment_counter,
                sample_count=transcribed_samples.shape[0],
                final=True
            )

            return

        transcribed_samples = np.concatenate((transcribed_samples, chunk), axis=0)
        if transcribed_samples.shape[0] / SAMPLING_RATE < settings.MIN_TRANSCRIBED_DURATION:
            continue

        # logfire.info(f'Processing audio with length {transcribed_samples.shape[0] / SAMPLING_RATE}s')
        output: List[Hypothesis] = model.transcribe(transcribed_samples, timestamps=True, verbose=False)

        transcribed_chain: List[Word] = output[0].timestamp['word']

        if len(transcribed_chain) == 0:
            continue

        if transcribed_chain[0]['word'].strip().lower() == last_confirmed_word.lower():
            transcribed_chain = transcribed_chain[1:]

        segments.append(transcribed_chain)

        # for seq_id, seq in enumerate(segments):
        #     logfire.info(f'{seq_id}: {" ".join([x["word"] for x in seq])}')
        #
        # logfire.info('...')
        best_segment_index, best_segment_length, best_segment_repetitions = -1, -1, -1
        for reference_segment_i in range(len(segments) - 1, 0, -1):
            ref_segment = segments[reference_segment_i]

            if len(ref_segment) < settings.STABLE_WORDS_PER_ITERATION:
                break

            reference_word_tuple = tuple([x['word'].lower().strip() for x in ref_segment])

            ref_seq_occurrences = 1
            matching_words = 0
            for comparison_segment_i in range(reference_segment_i - 1, 0, -1):
                comparison_segment = segments[comparison_segment_i]

                if len(comparison_segment) < settings.STABLE_WORDS_PER_ITERATION:
                    break

                comparison_segment_tuple = tuple([x['word'].lower().strip() for x in comparison_segment])

                i = 0
                while i < min(len(reference_word_tuple), len(comparison_segment_tuple)) and reference_word_tuple[i] == comparison_segment_tuple[i]:
                    i += 1

                if i >= settings.STABLE_WORDS_PER_ITERATION:
                    # logfire.warn(f'{reference_segment_i} == {comparison_segment_i}')
                    ref_seq_occurrences += 1

                    matching_words = i

            if ref_seq_occurrences >= settings.STABLE_ITERATION_COUNT:
                best_segment_index = reference_segment_i
                best_segment_length = matching_words
                break

        if best_segment_index != -1:
            best_s_words = segments[best_segment_index][:best_segment_length]
            logfire.info(f'Commiting! {best_segment_index} {[w["word"] for w in best_s_words]}')

            yield TranscribedSegment(
                complete=True,
                words=[w['word'] for w in best_s_words],
                start=current_ts_offset,
                id=stable_segment_counter,
                sample_count=transcribed_samples.shape[0]
            )

            samples_to_advance = int(best_s_words[-1]['end'] * SAMPLING_RATE)
            transcribed_samples = transcribed_samples[samples_to_advance:]
            segments = [segments[best_segment_index][best_segment_length:]]
            stable_segment_counter += 1
            current_ts_offset += samples_to_advance / SAMPLING_RATE

        else:
            yield TranscribedSegment(
                complete=False,
                words=[w['word'] for w in transcribed_chain],
                sample_count=transcribed_samples.shape[0],
                start=current_ts_offset
            )
