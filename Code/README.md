# Code

The longer term goal for this code repository is to have single file self contained easy to follow examples for key xNN applications that execute in a moderate amount of time on Google Colab.  In many cases this implies the need for downsampled data sets, transfer learning and various problem simplifications.  However, the intention is to use the same network structures and training methods as would be applied to larger problems, thus making the jump from class example to real world application straightforward with increased resources.

Specific applications include:

- Vision (focus: CNNs)
  - Image classification (ResNetV2)
  - Pixel classification
  - Object detection (Faster R-CNN with ResNetV2-50 and FPN)
  - Object segmentation (Mask R-CNN with ResNetV2-50 and FPN)
  - Depth estimation
  - Motion estimation

- Language (focus: self attention and attention)
  - Language modeling (BERT)
  - Question answering
  - Language to language translation (Transformer)

- Speech (focus: RNNs and attention)
  - Speaker identification
  - Keyword spotting and command recognition
  - Speech to text transduction (RNN Transducer)

- Games (focus: value, policy and model based methods)
  - Atari (DQN)
  - Go (MCTS with value and policy networks)

This is (obviously) a work in progress
