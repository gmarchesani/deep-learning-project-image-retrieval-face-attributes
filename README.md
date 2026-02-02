## Goal
Image retrieval refers to the process of finding images that are relevant to a given search query. Traditionally, this task has relied on metadata such as tags, annotations, or textual descriptions. However, more advanced systems perform retrieval based on the visual content of images themselves, enabling searches based on color, texture, shape, or even high-level semantic features.

In this project, we focus on implementing an attribute-based image retrieval system that can search for images of people based on a set of facial attributes. Each query is defined by a set of semantic attributes such as gender, hair color, or face shape. This problem naturally leads to a multimodal learning setting, where two different data types, images and their corresponding attribute vectors, must be aligned in a common representation space, allowing direct comparison between them.

## Implementation
To address the objective of attribute-based face image retrieval, we employed a deep metric learning framework with weak supervision, combining:
* a **Convolutional Neural Network - (CNN)** that generates embeddings from images
* a **Fully-Connected Neural Network (MLP)** that generates embeddings from attribute vectors.
  
For the image modality, we adopt a transfer learning strategy using a convolutional neural network pretrained on ImageNet (ResNet50) as a feature extractor, followed by a custom head that maps these features into the embedding space.\
In parallel, the attribute branch is implemented as a multilayer perceptron (MLP) composed of fully connected layers with non-linear activations.\
\
As a training strategy, we kept frozen the backbone of the image network, while we trained the embedding head from scratch using a weak supervision approach based on a batch-all triplet loss formulation. For each image embedding (anchor), the corresponding attribute embedding is treated as the positive, while other attribute embeddings in the batch serve as negatives.
Both image and attribute embeddings are learned jointly so that semantically corresponding images and attribute descriptions are close to each other, while mismatched pairs are pushed apart. 

After training, image retrieval is performed by embedding a query attribute vector and ranking all images in the retrieval set according to their similarity in the learned embedding space. This enables effective and flexible image retrieval based solely on semantic attribute queries.

## Dataset Access Instructions
As requested, the dataset is not included directly in the submission.\
Instead, the notebook provides a fully automatic and environment-aware loading procedure, ensuring that the CelebA dataset can be obtained easily on any platform.

There are two supported ways to obtain the data:
1. **Using the CelebA dataset from Kaggle**\
   If the notebook is executed on Kaggle, simply attach the official “CelebA Dataset” as an input dataset. \
   Download the images from this link: https://github.com/jhalmes/celeba \
   Download the annotations from this link: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&amp%3Busp=sharing \
   and then recreate the following structure in ***/kaggle/input/celeba/*** folder:\
   celeba\
├── img_align_celeba\
│   ├── 000001.jpg\
│   ├── 000002.jpg\
│   └── ...\
├── identity_CelebA.txt\
├── list_attr_celeba.txt\
├── list_bbox_celeba.txt\
├── list_eval_partition.txt\
└── list_landmarks_align_celeba.txt

   The notebook automatically detects the Kaggle environment and loads the dataset from:
   ***/kaggle/input/celeba/***
   
3. **Using the CelebA dataset via TorchVision (Colab or local)**\
   If the notebook is executed on Google Colab or on a local machine, the dataset is automatically downloaded through:\
   ***torchvision.datasets.CelebA(..., download=True)***


The notebook will store the downloaded files in:
* /content/celeba/ on Colab
* ./data/celeba/ when run locally
so that the dataset is only downloaded once.
