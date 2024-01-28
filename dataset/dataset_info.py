def ESD_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. 'DATAPATH/0018/Surprise/0018_001417.wav'
    
    #=== OUTPUT
    * Speaker Number    || int
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test
    * filename          || str
    """
    
    spk_num, emo_state, filename = pth_wav.split('/')[-3:]
    
    filename = filename.replace(".wav", ".npy")
    filename = "_".join([emo_state, filename])
    return spk_num, emo_state, filename
    
    
def EmovDB_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. "DIRNAME/bea/amused_1-15_0010.wav"
    
    #=== OUTPUT
    * Speaker Name      || str
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test'
    * file Number       || int
    """
    
    spk_name = pth_wav.split('/')[-2]
    emo_state = pth_wav.split('/')[-1].split('_')[0]
    file_number = int(pth_wav.split('/')[-1].split('_')[-1].replace('.wav', ''))
    
    if 1 <= file_number and 30 >= file_number:
        data_mode = 'test'
    elif 31 <= file_number and 50 >= file_number:
        data_mode = 'eval'
    else:
        data_mode = 'train'
        
    if emo_state == "amused":
        emo_state = "Amused"
    elif emo_state == "anger":
        emo_state = "Angry"
    elif emo_state == "neutral":
        emo_state = "Neutral"
    elif emo_state == "sleepiness":
        emo_state = "Sleepy"
    elif emo_state == "sleepiness":
        emo_state = "Sleepy"
    elif emo_state == "disgust":
        emo_state = "Disgusted"
    elif emo_state == "Disgust":
        emo_state = "Disgusted"
        
    file_name = "_".join([emo_state, spk_name, "{}".format(str(file_number).zfill(4)) + ".npy"])
    return spk_name, emo_state, file_name


def RAVDESS_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. "DIRNAME/Actor_04/03-01-02-02-01-01-04.wav"
    
    #=== OUTPUT
    * Speaker Name      || str
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test'
    * file Number       || int
    
    #=== Filename Identifiers
        - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        - Vocal channel (01 = speech, 02 = song).
        - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        - Repetition (01 = 1st repetition, 02 = 2nd repetition).
        - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
        
    ! Notice
    We only use the data having strong emotional intensity.
    """
    
    _dict_emoState = {
        '01': 'Neutral',
        '02': 'Calm',
        '03': 'Happy',
        '04': 'Sad',
        '05': 'Angry',
        '06': 'Fear',
        '07': 'Disgust',
        '08': 'Surprised',
    }
    
    _, _, emo_num, _, stt, rep, spk_num = pth_wav.split('/')[-1].replace('.wav', '').split('-')
    
    file_number = (int(stt) - 1) * 2 + int(rep)
        
    emo_state = _dict_emoState[emo_num]
    file_name = "_".join([emo_state, spk_num, "{}".format(str(file_number).zfill(4)) + ".npy"])
    return spk_num, emo_state, file_name
    
    
    

def SAVEE_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. "DIRNAME/DC_f15.wav"
    
    #=== OUTPUT
    * Speaker Name      || str
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test'
    * file Number       || int
    
    #=== Filename Identifiers
        - Emotion (a = Angry, d = Disgust, f = Fear, h = Happy, n = Neutral, sa = Sad, su = Surprise)
    """
    
    _dict_emoState = {
        'n': 'Neutral',
        'f': 'Fear',
        'h': 'Happy',
        'sa': 'Sad',
        'a': 'Angry',
        'd': 'Disgust',
        'su': 'Surprised',
    }
    
    file_name = pth_wav.split('/')[-1].replace('.wav', '')
    
    file_num = file_name[-2:]
    spk_name = file_name[:2]
    emo_state = _dict_emoState[file_name.split('_')[-1][:-2]]
    
    file_name = "_".join([emo_state, spk_name, file_num + ".npy"])
    return spk_name, emo_state, file_name

    
    
def JL_Corpus_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. "DIRNAME/female1_angry_5b_1.wav"
    
    #=== OUTPUT
    * Speaker Name      || str
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test'
    * file Number       || int
    
    #=== Filename Identifiers
        - Emotion (a = Angry, d = Disgust, f = Fear, h = Happy, n = Neutral, sa = Sad, su = Surprise)
    """

    _dict_emoState = {
        'neutral': 'Neutral',
        'happy': 'Happy',
        'sad': 'Sad',
        'angry': 'Angry',
        'excited': "Excited"
    }

    file_name = pth_wav.split("/")[-1]
    
    spk_name = file_name.split("_")[0]
    emo_state = _dict_emoState[file_name.split("_")[1]]
    file_num = file_name.split("_")[2] + '_' + file_name.split("_")[3].replace('.wav', '.npy')
    
    file_name = "_".join([emo_state, spk_name, file_num])
    return spk_name, emo_state, file_name
    
    
    
