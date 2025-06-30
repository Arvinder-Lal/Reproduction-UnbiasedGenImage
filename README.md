# Uncovering Biases in Deepfake Detection Datasets: A Reproduction of “Fake or JPEG?”
(Adapted from https://github.com/gendetection/UnbiasedGenImage and https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)

This repository builds upon the work of Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. That project investigates the biases in datasets for AI-generated image detection using the GenImage dataset.

# Modifications
Frequency Domain Processing: We implemented two classes in create_dataset.py to transform images into the frequency domain using Fast Fourier Transform (FFT) and Wavelet Transform.
Real-Time Deepfake Detection: We integrated a real-time detection module, adapted from public deepfake detection models. Just as the authors of Fake or JPEG? have done, we changed the dataset (dataset.py) to be more suitable for the experiments and added the  get_data.py for selecting the right data from the csv file and get_transform.py for transformations. 

# Unbiased GenImage dataset
Like the original authors, this project uses the GenImage dataset, along with the additional metadata CSV provided in their work.

⚠️ Important: Please refer to the original repository for instructions on how to download and prepare the GenImage dataset. 

1) Steps to Prepare Dataset:
Download GenImage dataset and metadata:
Follow the instructions in the original repository. They provide download scripts and preprocessing tools to clean corrupted files and extract a meaningful subset.

2) Optional: Create a size/compression-filtered subset:
Use their example code to filter images by size and compression quality, as shown in their paper.

 # Model Training & Evaluation
This codebase includes training and validation pipelines for ResNet50 and Swin-Transformer (Swin-T), consistent with the original paper. However, the dataset loading logic is modified to include optional frequency-domain preprocessing.

Training commands, configuration files, and detector logic remain largely aligned with the original implementation, with added support for frequency-domain analysis and can be found in the respective ReadMe of the detectors. To use FFTTransform during training add --use_fft and to use WaveletTransforn add --use_wavelet. 

## Results

### ResNet50

<p align="center">
  <img src="results/results_resnet.png" width="80%" />
  <br>
  <em>Cross-Generator Performance when training ResNet50 on constrained dataset</em>
</p>

<br>

<p align="center">
  <img src="results/results_resnet_diff.png" width="80%" />
  <br>
  <em>Difference to when training on raw dataset</em>
</p>

<br><br>

### Swin-T

<p align="center">
  <img src="results/results_swin.png" width="80%" />
  <br>
  <em>Cross-Generator Performance when training Swin-T on constrained dataset</em>
</p>

<br>

<p align="center">
  <img src="results/results_swin_diff.png" width="80%" />
  <br>
  <em>Difference to when training on raw dataset</em>
</p>
