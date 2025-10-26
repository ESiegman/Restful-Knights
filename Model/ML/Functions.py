import mne
import numpy as np
from scipy import signal
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim

# Parameters based on data
SAMPLING_RATE = 100  # (fs) Hz
EPOCH_DURATION = 30  # (seconds) - Standard for sleep staging
NPERSEG = 5 * SAMPLING_RATE  # (500 samples) 2-second sub-window for STFT
NOVERLAP = NPERSEG // 2  # (250 samples) 50% overlap
IMG_SIZE = (128, 128)  # (height, width) Final image size for the CNN
stage_map = {'Sleep stage W': 0,   # Wake
             'Sleep stage 1': 1,   # N1
             'Sleep stage 2': 2,   # N2
             'Sleep stage 3': 3,   # N3
             'Sleep stage 4': 3,   # N3 (Combining N4 into N3)
             'Sleep stage R': 4,   # REM

             # Not used in final data usage, but needed for data processing
             'Sleep stage ?': -1} # describes each sleep stage
NUM_CLASSES = 5 # 5 sleep stages
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD & FILTER DATA
# This function loads, filters, and selects EEG channel.

##
# @brief Loads and filters the EEG data from one channel.
# @param file_path Path to the EEG data file.
# @param channel Name of the EEG channel to select.
# @return Tuple containing filtered EEG data and raw MNE object.
def load_and_filter_eeg(file_path, channel='EEG Fpz-Cz'):
    eeg_file_path = "SC4001E0-PSG.edf"
    if eeg_file_path:
        # load the file
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True)

        # select EEG channel
        raw.pick_channels([channel]) # fpz - cz (center forehead to ground)

        # -------- FILTERING THE DATA
        # Band-pass filter to keep only relevant sleep frequencies
        # Reduces anything super-small and anything too big
        raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin')

        # Notch filter to remove 60 Hz power line noise
        # raw.notch_filter(freqs=60.0, fir_design='firwin')

        # Get the raw data and sampling rate
        eeg_data = raw.get_data()[0]  # Get data from the one channel
        fs = int(raw.info['sfreq']) # 100 Hz

        # Ensure our parameters match the file's
        global SAMPLING_RATE, NPERSEG, NOVERLAP
        SAMPLING_RATE = fs
        NPERSEG = 2 * SAMPLING_RATE
        NOVERLAP = NPERSEG // 2

        return eeg_data, raw

    else:
        # --- FOR A FAKE (SIMULATED) SIGNAL ---
        print("No file path given, using simulated EEG data.")
        n_samples = SAMPLING_RATE * 60 * 10  # 10 minutes of data
        # Create a mix of delta (2Hz), alpha (10Hz), and spindle (13Hz)
        times = np.arange(n_samples) / SAMPLING_RATE
        delta = np.sin(2 * np.pi * 2 * times)
        alpha = np.sin(2 * np.pi * 10 * times)
        spindle = np.sin(2 * np.pi * 13 * times)
        noise = np.random.randn(n_samples) * 0.5
        # Simulate different stages
        eeg_data = np.concatenate([
            (delta + noise)[:n_samples // 3],
            (alpha + noise)[n_samples // 3: 2 * n_samples // 3],
            (spindle + noise)[2 * n_samples // 3:]
        ])
        return eeg_data

##
# @brief Converts MNE annotations into an array of integer labels, where each label corresponds to a 30-second epoch.
# @param annotations MNE annotations object.
# @param stage_map Dictionary mapping stage names to integer codes.
# @param epoch_duration Duration of each epoch in seconds.
# @param raw MNE raw EEG object.
# @return Array of integer labels for each epoch.
def create_y_labels(annotations, stage_map, epoch_duration, raw):
    stage_names = list(stage_map.keys())

    # filter annotations to include only the stages we care about
    stages_annotations = annotations[np.isin(annotations.description, stage_names)]

    # convert annotations to events
    events, event_id = mne.events_from_annotations(raw, event_id=stage_map)

    # determine the total number of epochs (based on the end time)
    # The last event time is the last row's start time + duration.
    last_event_end = events[-1, 0] * epoch_duration + annotations.duration[-1]

    # Total number of 30s epochs
    n_epochs = int(np.ceil(last_event_end / epoch_duration))

    # Initialize the Y label array with -1 to account that some stages were not labeled
    y_labels = np.full(n_epochs, -1)

    # fill the Y label array
    # Initialize indexer for stages_annotations
    stage_idx = 0

    # fill the Y label array
    for onset, _, stage_code in events:
        # get the current duration from the stages_annotations
        stage_duration = stages_annotations.duration[stage_idx]
        # update counter
        stage_idx += 1

        # onset is in samples, where 1 sample = 30 seconds
        start_epoch_index = onset

        # Figure out how many 30-second epochs the stage event covers
        end_epoch_index = start_epoch_index + int(np.ceil(stage_duration / epoch_duration))

        # Assign the stage code to the corresponding epoch indices
        y_labels[start_epoch_index:end_epoch_index] = stage_code

    # Remove any un-scored epochs (-1)
    y_labels_final = y_labels[y_labels != -1]

    return y_labels_final

##
# @brief Chops the data into 30s epochs.
# @param data EEG data array.
# @param epoch_len_samples Number of samples per epoch.
# @return Array of epochs.
def create_epochs(data, epoch_len_samples):
    num_epochs = len(data) // epoch_len_samples # obtain total epochs
    epochs = []

    for i in range(num_epochs):
        start = i * epoch_len_samples # start of data range
        end = start + epoch_len_samples # end (30s length)
        epochs.append(data[start:end])
    return np.array(epochs)

##
# @brief Converts a single epoch (1D array) into a 2D spectrogram image.
# @param epoch_data 1D array of EEG data for one epoch.
# @return 2D spectrogram image.
def create_spectrogram_image(epoch_data):
    # Calculate Short-Time Fourier Transform (STFT)
    frequencies, times, Zxx = signal.stft(
        epoch_data,
        fs=SAMPLING_RATE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP
    )

    # Get magnitude and convert to decibels (log scale)
    # Adding 1e-9 avoids log(0) errors
    log_spectrogram = 10 * np.log10(np.abs(Zxx) + 1e-9)

    # Filter to relevant frequencies
    # We set our filter to 40 Hz, but STFT might return up to fs/2.
    freq_mask = (frequencies >= 0.5) & (frequencies <= 40)
    log_spectrogram = log_spectrogram[freq_mask, :]

    # Resize the image to size for the CNN
    resized_spectrogram = cv2.resize(
        log_spectrogram,
        IMG_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    return resized_spectrogram

##
# @brief Normalizes a single image to be Z-score (mean=0, std=1).
#        This helps the CNN learn patterns, not just brightness.
# @param image 2D image array.
# @return Normalized image.
def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:  # Avoid division by zero
        return image - mean
    return (image - mean) / std

##
# @class EEGDataset
# @brief Custom PyTorch Dataset for EEG data and labels.
class EEGDataset(Dataset):
    ##
    # @brief Constructor for EEGDataset.
    # @param X Input data.
    # @param Y Labels.
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    ##
    # @brief Returns the number of samples in the dataset.
    # @return Number of samples.
    def __len__(self):
        return len(self.Y)

    ##
    # @brief Returns the data and label at the given index.
    # @param idx Index.
    # @return Tuple (data, label).
    def __getitem__(self, idx):
        # We use the integer label here. CrossEntropyLoss will handle the one-hot internally.
        return self.X[idx], self.Y[idx]

##
# @class CNN
# @brief Convolutional Neural Network for sleep stage classification.
class CNN(nn.Module):
    ##
    # @brief Constructor for CNN.
    # @param in_channels Number of channels in the input image.
    # @param num_classes Number of output classes.
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        # Sequential Model for the sequential nature of the problem
        self.features = nn.Sequential(
            # 1 -> 32 filters, 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 32 -> 64 filters, 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 64 -> 128 filters, 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Input: 1x128x128 -> Output: 128x16x16
        # Flat size = 128 * 16 * 16 = 32,768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)  # Final output layer for 5 sleep stages
        )

    ##
    # @brief Forward pass of the CNN.
    # @param x Input tensor.
    # @return Output tensor.
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

##
# @brief Trains the EEG classification model.
# @param model PyTorch model to train.
# @param train_loader DataLoader for training data.
# @param val_loader DataLoader for validation data.
# @param num_epochs Number of epochs to train.
def train_eeg_model(model, train_loader, val_loader, num_epochs):
    lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # define trackers
    best_val_loss = float('inf')
    patience_counter = 0 # number of epochs that it runs until after loss stops decreasing
    patience = 5

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        model.train()  # Set model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)
            loss = lossfunc(outputs, labels) # calculate loss for classification
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Evaluate the validation --------------------------------
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = lossfunc(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # calculate the metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        # print metrics
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")



##
# @brief Tests the EEG classification model.
# @param model PyTorch model to test.
# @param test_loader DataLoader for test data.
# @param device Device to run the model on.
# @return Test accuracy as a percentage.
@torch.no_grad()
def test_eeg_model(model, test_loader, device):

    model.eval()

    # keep track of values
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        # Move data to desired device
        inputs, labels = inputs.to(device), labels.to(device)

        # Get outputs
        outputs = model(inputs)

        # Get classes with highest scores
        # torch.max returns (max_value, max_index)
        _, predicted = torch.max(outputs.data, 1)

        # Update counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # calculate final metrics and print
    accuracy = 100 * correct / total
    print(f"\n--- Final Test Results ---")
    print(f"Test Accuracy on {total} epochs: {accuracy:.2f}%")
    print("--------------------------")
    return accuracy

if __name__ == "__main__":
    import torch._inductor.config
    torch._inductor.config.max_autotune_gemm_backends = "TRITON"

    # Load data
    filtered_eeg_data, raw_eeg_data = load_and_filter_eeg(file_path=None, channel='EEG Fpz-Cz')

    # Create the epochs
    epoch_len_in_samples = EPOCH_DURATION * SAMPLING_RATE
    eeg_epochs = create_epochs(filtered_eeg_data, epoch_len_in_samples)

    print(f"Loaded and filtered data. Found {len(eeg_epochs)} epochs.")

    # Process each epoch
    X_images = []
    for epoch in eeg_epochs:
        spectrogram_img = create_spectrogram_image(epoch)
        normalized_img = normalize(spectrogram_img)
        X_images.append(normalized_img)

    # Read hypnograph files that store data about labels and stages of sleep
    hypno_file_path = "SC4001EC-Hypnogram.edf"
    annotations = mne.read_annotations(hypno_file_path)

    raw_eeg_data.set_annotations(annotations)

    # obtain the eeg data aligned annotations
    aligned_annotations = raw_eeg_data.annotations

    # Stack all images into a single NumPy array
    # This is the final data ready for the CNN
    X = np.array(X_images)
    Y_labels = create_y_labels(aligned_annotations, stage_map, EPOCH_DURATION, raw_eeg_data)

    # Convert the integer labels to one-hot encoded vectors
    Y_train_one_hot = torch.nn.functional.one_hot(torch.from_numpy(Y_labels), num_classes=NUM_CLASSES)

    # Add the "channels" dimension (1 for grayscale)
    X = X[..., np.newaxis]

    # (Number of Epochs, Image Height, Image Width, 1)
    print(f"Final data shape for CNN: {X.shape}")
    print(f"Final label shape for CNN: {Y_labels.shape}")

    # ---------- STARTING THE TRAINING / TESTING SPLIT

    # Make the variables pytorch friendly (Batch, Channels, Height, Width)
    X_tensor = torch.from_numpy(X).float().permute(0, 3, 1, 2)
    Y_tensor = torch.from_numpy(Y_labels).long()

    # ----------------- Split Data
    # train, test
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X_tensor, Y_tensor, test_size=0.15, random_state=42, stratify=Y_tensor
    )

    # Split train and validation
    X_train_split, X_val, Y_train_split, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.20, random_state=42, stratify=Y_temp
    )

    # Now I need to create DataLoaders
    BATCH_SIZE = 64

    # process each data set
    train_dataset = EEGDataset(X_train_split, Y_train_split)
    val_dataset = EEGDataset(X_val, Y_val)
    test_dataset = EEGDataset(X_test, Y_test)

    # Put into data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Total epochs reserved for final testing: {len(test_dataset)}")

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # create model
    # in_channel`s = 1 for grayscale
    model = CNN(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model = torch.compile(model)

    # Call training function!
    train_eeg_model(model, train_loader, val_loader, num_epochs=32)

    test_eeg_model(model, test_loader, device)

##
# @brief Summarizes sleep stages from y_labels and writes results to a CSV.
# @param y_labels Array of sleep stage labels.
# @param output_csv Output CSV file path.
def summarize_eeg_to_csv(y_labels, output_csv="EEG_results.csv"):
    import numpy as np
    import csv

    # Calculate durations
    total_epochs = len(y_labels)
    stage_counts = {stage: np.sum(y_labels == stage) for stage in range(5)}
    epoch_duration = 30  # seconds

    results = [
        ["Total Epochs", total_epochs],
        ["Total REM Duration (min)", stage_counts[4] * epoch_duration / 60],
        ["Total Deep Sleep (N3) Duration (min)", stage_counts[3] * epoch_duration / 60],
        ["Number of Awakenings", stage_counts[0]],
        ["N1 Duration (min)", stage_counts[1] * epoch_duration / 60],
        ["N2 Duration (min)", stage_counts[2] * epoch_duration / 60],
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(results)
