# Time Series Monitoring Framework using Image Encoding Techniques

## Abstract
Monitoring time series is considered highly valuable for detecting possible anomalies. This ability is relevant in numerous application domains, including health and industry, and many applications, such as monitoring system failures, detecting external cyber-attacks, and diagnosing diseases. The most common models for time series data exploit recurrent neural network (RNN) architecture. However, such architecture requires a high computational cost and shows some problems, such as vanishing and exploding gradients. Inspired by the success of computer vision methods, several studies have proposed transforming time series into images by applying encoding time series algorithms that can be analyzed by deep-learning models based on convolutional neural networks (CNN).

This work proposes a framework for monitoring time series using image encoding techniques, comparing the performance of various encoding techniques and CNN architectures for time series classification tasks. Additionally, an alternative implementation of the Recurrence Plot, incorporating Dynamic Time Warping (DTW), is introduced. This approach eliminates the need to resample time series with different frequencies.

The performance of four encoding techniques—Gramian Angular Field (GAF), Markov Transition Field (MTF), Recurrence Plot (RP), and Recurrence Plot with Dynamic Time Warping (RP-DTW)—is evaluated using three CNN architectures: a custom CNN, a modified VGG16, and a modified ResNet18. The evaluation is conducted on the WESAD (Wearable Stress and Affect Detection) dataset, which contains physiological and motion data for stress and affect detection.

The best-performing combination was the VGG16 architecture with the GAF encoding method, achieving an accuracy of 95.8% and a weighted F1-score of 93.17%. Although the RP-DTW approach did not achieve the highest performance, it demonstrated potential and highlighted intriguing possibilities for future research. Its novel aspects suggest that with further investigation and refinement, RP-DTW could make significant contributions to the field.

## Dataset
The dataset used in this project is the WESAD (Wearable Stress and Affect Detection) dataset, which contains physiological and motion data for stress and affect detection. To use the dataset in this project, download it and place it into a directory called `data` in the project.

You can download the WESAD dataset by following the instructions on this [link](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).

## Installation
To install the required packages, you can use the `requirements.txt` file. Run the following command:
```bash
pip install -r requirements.txt
```
## Running the Code
You can run the main script with various options. Here is an example of how to use the script:
```bash
python main.py --freq 700 --dataset data/WESAD --sec 60 --window_stride 1 --patience 10 --epochs 100 --methods gaf_difference gaf_summation mtf rp_euclidean rp_dtw --model custom --labels 1 2
```

### Command Line Arguments
- `--freq`: Frequency of the preprocessing (default: 700)
- `--dataset`: Path to the dataset (default: 'data/WESAD')
- `--sec`: Window in seconds (default: 60)
- `--window_stride`: Window stride in seconds (default: 1)
- `--patience`: Early stopping patience (default: 10)
- `--epochs`: Number of epochs (default: 100)
- `--methods`: Methods to run (default: ['gaf_difference', 'gaf_summation', 'mtf', 'rp_euclidean', 'rp_dtw'])
- `--model`: Model choice for training (choices: ['custom', 'vgg', 'resnet'], default: 'custom')
- `--labels`: List of labels for classification (default: [1, 2])

## Project Structure
- `preprocessing.py`: Contains functions for preprocessing the data.
- `encoding/`: Contains the different encoding techniques (GAF, MTF, RP, RP-DTW).
- `model/`: Contains the CNN models used in this project.
- `utils/`: Contains utility functions for email notifications, plotting, etc.
- `main.py`: The main script for running the experiments.

## Results
The results of the experiments are saved in the `Results/` directory. Plots and model checkpoints are saved for each experiment run.

## Contact
For any questions or issues, please contact Marco Serenelli on Github or Linkedin.
