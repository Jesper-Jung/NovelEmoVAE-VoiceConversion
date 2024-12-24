# Novel VAE-based Framework to Infer Complex Speaking Style from Arbitrary Speaker and Emotion State : 임의의 화자 및 감정 상태로부터 복잡한 발화 스타일 추론을 위한 새로운 VAE 기반 프레임워크

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
   - [model_weights.pth](https://path-to-model-weights)
2. **Training Dataset**:
   - [ESD Dataset](https://path-to-dataset)
   - [EmovDB Dataset](https://path-to-dataset)
  project/ ├── data/ │ ├── train/ │ │ └── training_data.json │ ├── val/ │ │ └── validation_data.json └── weights/ └── model_weights.pth
4. **Validation Dataset**:
3. **Example Input Data**:
   - [input_data.json](https://path-to-example-data)
