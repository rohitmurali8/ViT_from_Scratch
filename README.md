# ViT_from_Scratch

This repository contains an implementation of Vision Transformers (ViT) from scratch using PyTorch. The goal is to help users understand how Vision Transformers work by building the model from scratch and training it on image classification tasks. Along with building ViT model architecture from scratch, the repository also goes into detail about the math involved inside each of these blocks.   

### Model Architecture


### Project Overview

This repository implements Vision Transformer model from scratch on the MNIST dataset to perform image classification. This repository provides a PyTorch-based implementation of ViT, covering the step-by-step components. Vision Transformers (ViT) are a deep learning architecture that applies the transformer model (originally developed for NLP) to image data.


The Vision Transformer model consists of the following components:

- **Patch Embedding**: Breaking images into fixed-size patches and converting them into a flat vector representation. This is done by simply reshaping the input, which has size (N, C, H, W) (in our example (N, 1, 28, 28)), to size (N,  Number of Patches, Patch dimensionality), where the dimensionality of a patch is adjusted according to the number of patches choosen. Here, N represents the number of images inside the batch. The formula for the output shape after the Patchify function is given as: (N, P², HWC/P²). So if we select the number of patches to be 7 along the height and width, we get the following output shape for the current batch of training data (N, 7x7, 4x4) = (N, 49, 16). The patches which are now flattened are then mapped through a linear layer. While each patch was 4X4 dimensions, the linearly mapped patches can be of any dimension. The linear mapping only happens with the last dimension of the patched images.

- **Classification Token**: This is a token that is stacked onto the output of the linearly mapped patches. This token captures information about the other tokens present in the other patches that are being sent to the encoder. When we have the information about all the tokens, we can perform image classification just using the token associated. The initial value of this token is a parameter of the model that needs to be learned. If we had another downstream task using this transformer, like classifying digits greater or less than 5 in the MNIST dataset, then we would just another token and corresponding output linear layers. The input dimensions after adding the classification token become (N, 50, 8). 

- **Positional Encoding**: Positional encoding allows the model to understand where patch in the image is placed. The positional encoding value is added to the input which is patchified, linearly mapped and padded with the classification token before passing it to the transformer encoder. It is suggested to have sine and cosine wave values as positional embeddings. The index of the patches or words in a sequence are not used as positional embeddings because depending on the length of the sequence those values can become arbitralily large. Positional encoding adds high-frequency values to the first dimensions and low-frequency values to the latter dimensions. This positional embedding is a function of the number of elements in the sequence and the dimensionality of each element. Thus, it is always a 2-dimensional tensor or “rectangle”. Here’s a simple function that, given the number of tokens and the dimensionality of each of them, outputs a matrix where each coordinate (i,j) is the value to be added to token i in dimension j.In each sequence, for token i we add to its j-th coordinate the following value:

![Representation of the poositional encoding values]()

- **Multi-Headed Self Attention (MHA)**: For a single image, each patch needs to be updated based on some similarity measure with the other patches. We do so by linearly mapping each patch (that is now an 8-dimensional vector in our example) to 3 distinct vectors: q, k, and v (query, key, value). Then, for a single patch, we are going to compute the dot product between its q vector with all of the k vectors, divide by the square root of the dimensionality of these vectors, softmax these so-called attention cues, and finally multiply each attention cue with the v vectors associated with the different k vectors and sum all up. In this way, each patch assumes a new value that is based on its similarity (after the linear mapping to q, k, and v) with other patches. This whole procedure, however, is carried out H times on H sub-vectors of our current 8-dimensional patches, where H is the number of Heads.

- **Transformer Encoder**: Applying standard transformer layers to capture spatial and contextual relationships. Once we form the input transformations in the previous steps, the encoder layer is formed by a combination of Layer Normalization layer, MHA layers along with residual connections between the outputs of these layers as shown in the encoder block in the model architecture figure. 

- **Classification Head**: A final fully connected layer for image classification. The classification token (first token) is extracted out of our N sequences, and each token is used to get N classifications. Since we fixed each token to be an 8-dimensional vector, and since there are 10 possible digits, we can implement the classification MLP as a simple 8x10 matrix, activated with the SoftMax function.

### Results and Usage

- The model is trained for about 5 epochs with a learning rate of 0.005. The model weights are optimized using the Adam optimizer with the loss function being Cross Entropy loss. 
- The hidden dimension of the model is split to use 2 heads and we use 2 encoder blocks from the transformer architecture.
- The vision transformer model achieves an accuracy of 83% when it is trained for just 5 epochs from scratch. 

In this repository 
This repository offers a clean, modular, and commented implementation to make it easy to follow and modify for your own use cases.
Features

    Vision Transformer Architecture: Implements the ViT model from scratch using PyTorch.
    End-to-End Image Classification Pipeline: Training and evaluation on datasets like CIFAR-10 or ImageNet.
    Flexible Hyperparameters: You can modify model parameters (like patch size, transformer layers, etc.) for experimentation.
    Clear Documentation: Detailed explanations and comments in the code to help you understand each component of the Vision Transformer model.

Requirements

    Python 3.x
    PyTorch >=1.10.0
    torchvision
    numpy
    matplotlib
    tqdm

You can install the necessary Python packages using pip:

pip install torch torchvision numpy matplotlib tqdm

Setup & Installation

    Clone this repository:

git clone https://github.com/rohitmurali8/vision-transformers-from-scratch-pytorch.git
cd vision-transformers-from-scratch-pytorch

    Install the dependencies:

pip install -r requirements.txt

References

    Original Vision Transformer paper
    PyTorch documentation
    Medium guide by Brian Pulfer

License

This project is licensed under the MIT License - see the LICENSE file for details.
