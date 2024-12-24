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
   - [checkpoint_best_legacy_500.pt]([https://path-to-model-weights](https://github.com/auspicious3000/contentvec?tab=readme-ov-file))
2. **Training Dataset**:
   - [ESD Dataset](https://path-to-dataset)
   Please follow this structure for the ESD dataset directory.
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

   - [EmovDB Dataset](https://path-to-dataset)
   Please follow this structure for the EmovDB dataset directory.
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
   The speakers of the EmovDB Dataset are bea, jenie, josh and sam.

3. **Validation Dataset**:
   - [JL_Corpus Dataset](https://path-to-dataset)
   

  
4. **Example Input Data**:
   - [input_data.json](https://path-to-example-data)
