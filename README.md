# Novel VAE-based Framework to Infer Complex Speaking Style from Arbitrary Speaker and Emotion State

**ABSTRACT**

In the field of Voice Conversion aimed at conveying the emotional speaking style of the target speaker, many previous studies have opted to disentangle the speech features into several latent spaces containing only specific attributes such as content information or speaker identity. However, often fails to accurately reflect the identity of the target speaker or emotional states, and it struggles to adequately represent the complex characteristics of speaking styles, leading to poor quality in voice conversion. In contrast to conventional methods that separate features, it is crucial to effectively model a latent space that encompasses various speaker-specific speech characteristics and emotional expressions. This approach aims to reduce the training burden on the decoder to reflect complex nature of speaking style, while enabling the efficient and natural generation of speech that is not only more expressive but also rich in emotional nuances. This study introduces a novel VAE-based framework capable of modeling a latent space from speaker information and an emotional state, which consistently transfers any speaker identity along with its emotional speaking style of a given categorical emotion state in the converted speech, even with a simple model. The Adain-VC model is used as a posterior network capable of capturing speaking styles, and to effectively capture the distribution of speaking styles formed by this posterior network, it is employed in conjunction with a prior network composed of Normalizing Flows. To validate whether this latent space can effectively facilitate a complex emotional speaking style in converted speech, this framework was tested with two tasks - zero-shot Voice Conversion and Emotional Voice Conversion - conducted simultaneously.


## Experimental Environment
The following environment was used to train and test the model:
- **OS**: Ubuntu 20.04
- **GPU**: NVIDIA RTX 4090 / H100 NVL (CUDA 12.1)
- **PyTorch**: 2.1.0
- **Image Template**:
https://hub.docker.com/layers/pytorch/pytorch/2.1.0-cuda12.1-cudnn8-devel/images/sha256-fe174e1e257d29976c99ebe9832d9bb20bd9706ea8eff1482cc9af261998c48d


## Prerequisites
Please download the following files before running the project:

1. **Pre-trained Model Weights (ContentVec)**:
   Download the pre-trained model file of ContentVec.
   - [checkpoint_best_legacy_500.pt](https://github.com/auspicious3000/contentvec?tab=readme-ov-file)

2. **Training Dataset (ESD)**:
   - [ESD Dataset](https://github.com/HLTSingapore/Emotional-Speech-Data)
   Please follow this structure for the ESD dataset directory after you downloaded.
```
└───ESD
    ├───0011
    │   ├───Angry
    │   │   ├───0011_000351.wav
    │   │   ├───...
    │   │   └───0011_000700.wav
    │   ├───...
    │   └───Surprise
    │   ...
    └───0020
        ├───Angry
        ├───...
        └───Surprise
```  

3. **Training Dataset (EmovDB)**:
   - [EmovDB Dataset](https://openslr.org/115/)
   Please follow this structure for the EmovDB dataset directory after you downloaded.
```
└───EmovDB
    ├───bea
    │   ├───anger_1-28_0001.wav
    │   ├───...
    │   └───neutral_337-364_0364.wav
    │   ...
    └───sam
        ├───anger_1-28_0001.wav
        ├───...
        └───neutral_477-504_0504.wav
```
   .wav files below the name folder should have only emotional speeches of anger, neutral, and happy.
   <br>Amused and Sleepiness speeches are excluded to match the emotions with the ESD dataset and JL Corpus dataset.
   <br>Notice that the speakers of the EmovDB Dataset are bea, jenie, josh and sam.<br>

4. **Validation Dataset**:
   - [JL_Corpus Dataset](https://www.kaggle.com/datasets/tli725/jl-corpus)
   Please follow this structure for the EmovDB dataset directory after you downloaded.
```
└───JL_Corpus
    ├───female1_angry_1a_1.wav
    ├───female1_angry_1a_2.wav
    │   ...
    │
    │   ...
    └───male2_sad_15b_2.wav
```
   It is okay to contain .txt files and another any emotions such as anxious, encouraging, ...

5. **Setting Dataset Directory**
   <br>After downloading the datasets as described in [2. Training Dataset] and [3. Validation Dataset], organize the files into a folder named Dataset with the following structure:
```
└───Dataset
    ├───ESD
    ├───EmovDB
    └───JL_Corpus
```
   Then, make empty folders in the Dataset folder which named of "ESD_preprocessed", "EmovDB_preprocessed", and "JL_Corpus_preprocessed". <br>The final directory should look like this:
```
└───Dataset
    ├───ESD
    ├───ESD_preprocessed
    ├───EmovDB
    ├───EmovDB_preprocessed
    ├───JL_Corpus
    └───JL_Corpus_preprocessed
```
   
## Installation
Follow these steps to set up the project:

### Step 1. Clone the repository:
Download the project files to your local machine by running:

```bash
git clone https://github.com/Jesper-Jung/NovelEmoVAE-VoiceConversion.git
```

### Step 2. Install dependencies:
Install the required libraries and packages by running:

```bash
pip install -r requirements.txt
```

### Step 3. Configure in the YAML file:
- Open the configuration file (config.yaml, config_preprocessed.yaml).
- Then, update the pre-trained model path (config.yaml) and the dataset paths that you created earlier (config_preprocess.yaml) as described in prerequisites.
- Ensure the paths are consistent with the paths.

***config.yaml***
```yaml
  Pretrained:
    HuBERT:
      dense_model_name: "hubert-base-ls960"
      quantizer_name: "kmeans"
      vocab_size: 100

    ContentVec:
      model_path: "./EmoVAE/checkpoint_best_legacy_500.pt" ## Verify this to make consistent with your file.
```

***config_preprocess.yaml***
```yaml
Root_Dataset: "/workspace/Dataset"   ## Verify this to make consistent with your Dataset folder.

##############
# (Continue) #
##############

# Check in three section (ESD, EmovDB, and JL_Corpus)

  Paths:
    path_dataset: "/ESD"   # Verify this to make consistent with your ESD folder.
    path_save: "/ESD_preprocessed"   # Verify this to make consistent with your ESD_preprocessed folder.
```


## Installation
After setting the model parameters in the configuration file (config.yaml), then do this!

```bash
python3 train.py
```


## Conversion
```bash
python3 synthesizer.py --model_directory your_dir --saved_model your_model_pth
```

## Demo
A demo file is available in this repository.
