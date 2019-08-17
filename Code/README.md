# Code

The longer term goal for this repository is to have single file self contained easy to follow examples for key xNN applications that execute in a moderate amount of time on Google Colab and include code for data, encoder, decoder, error, update, evaluation and display.  In many cases this implies the need for downsampled data, transfer learning and problem simplification.  However, the intention is to use the same network structures and training methods as would be applied to larger problems, thus making the jump from class example to real world application straightforward with increased resources.

Target applications (and networks) include:

- Vision (focus: CNNs)
  - Image classification (ResNetV2)
  - Pixel classification
  - Object detection (Faster R-CNN with ResNet-50 V2 and FPN)
  - Object segmentation (Mask R-CNN with ResNet-50 V2 and FPN)
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
