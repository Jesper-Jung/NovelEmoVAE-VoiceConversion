from .datasetLoader import *

Codebook_EmoState = {
    "neutral": 0,       # ESD, Emov-DB
    "Neutral": 0,       # ESD
    "angry": 1,         # ESD, Emov-DB
    "Angry": 1,
    "happy": 2,         # ESD
    "Happy": 2,
    "Amused": 2,        # Emov-DB
    "sad": 3,           # ESD
    "Sad": 3,
    "Excited": 4,       # JL_Corpus
    "Surprised": 4,
    "Surprise": 4,      # ESD
    "Disgusted": 5,     # Emov-DB
    "Sleepy": 3         # Emov-DB
}


__all__ = ['Codebook_EmoState']