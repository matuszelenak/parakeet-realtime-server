import asyncio
import logging

from transcriber import continuous_transcriber

logging.disable(logging.CRITICAL)
import scipy.io.wavfile as wav
import nemo.collections.asr as nemo_asr


model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")


async def main():
    sample_rate, audio_data = wav.read('gn.wav')
    total_samples = audio_data.shape[0]

    async def samples_generator():
        for prefix_len in range(0, total_samples, 2000):
            yield audio_data[prefix_len:prefix_len + 2000]

    async for segment in continuous_transcriber(model, samples_generator()):
        if segment['complete']:
            print(segment['words'])
            # print(' '.join(segment['words']))

if __name__ == '__main__':
    asyncio.run(main())