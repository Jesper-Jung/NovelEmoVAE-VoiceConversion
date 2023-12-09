""" See, 
https://github.com/gallilmaimon/DISSC/tree/main
https://github.com/facebookresearch/textlesslib     "The Textless NLP project, https://speechbot.github.io/"
https://colab.research.google.com/github/facebookresearch/textlesslib/blob/main/examples/resynthesis_and_continuation.ipynb
"""

import torchaudio
from textless.data.speech_encoder import SpeechEncoder

dense_model_name = "hubert-base-ls960"
quantizer_name, vocab_size = "kmeans", 100
input_file = "./example_wav/input.wav"

# now let's load an audio example
waveform, sample_rate = torchaudio.load(input_file)
print(waveform.shape)       # torch([1, 88097])

# We can build a speech encoder module using names of pre-trained
# dense and quantizer models.  The call below will download
# appropriate checkpoints as needed behind the scenes. We can
# also construct an encoder by directly passing model instances
encoder = SpeechEncoder.by_name(
    dense_model_name=dense_model_name,
    quantizer_model_name=quantizer_name,
    vocab_size=vocab_size,
    deduplicate=False,
).cuda()


# now convert it in a stream of deduplicated units (as in GSLM)
print(waveform.shape)
encoded = encoder(waveform.cuda())
# encoded is a dict with keys ('dense', 'units', 'durations').
# It can also contain 'f0' if SpeechEncoder was initialized
# with need_f0=True flag.
units = encoded["units"]  # tensor([71, 12, 57, ...], ...)
print(encoded.keys())
print(units)


# encoder2 = SpeechEncoder.by_name(
#     dense_model_name=dense_model_name,
#     quantizer_model_name=quantizer_name,
#     vocab_size=vocab_size,
#     deduplicate=False,
# ).cuda()



# encoded2 = encoder(waveform.cuda())
# print(encoded2["units"].shape)