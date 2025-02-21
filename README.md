Device Configuration: The model will utilize CUDA if a GPU is available, otherwise it will use the CPU.
Data Preprocessing: It uses StandardScaler to standardize the features and convert them into PyTorch tensors.
Neural Network Architecture: A simple feed-forward neural network with one hidden layer.
Training: The model is trained for 100 epochs using the Adam optimizer and binary cross-entropy loss.
Evaluation: The model's accuracy is evaluated on both the training and test datasets.
