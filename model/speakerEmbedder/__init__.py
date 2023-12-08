kwarg_SPEAKER = {
    'model': 'ResNetSE34V2',
    'optimizer': 'adam',
    'trainfunc': 'softmaxproto',
    'nPerSpeaker': 1,
    'log_input': True,
    'encoder_type': 'ASP',
    'n_mels': 64,
    'eval_frames': 400,
    'nOut': 512,
    'nClasses': 5994
}