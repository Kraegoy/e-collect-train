# Load necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
existing_actions = np.array(['hello', 'thank you', 'how are you', "I'm fine", 'thank you'])
no_sequences = 1000
sequence_length =23

existing_model_path = 'action.h5'

if os.path.exists(existing_model_path):
    model = load_model(existing_model_path)
else:
    # Define a new model if no existing model is found
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(23, 1662)))  
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(existing_actions.shape[0], activation='softmax'))

# Detect new actions dynamically from directory structure
actions = [action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))]

# Update label map to include both existing and new actions
existing_actions = np.unique(np.concatenate((existing_actions, actions)))
all_actions = np.concatenate((existing_actions, actions))
label_map = {label: num for num, label in enumerate(all_actions)}


# Load and preprocess new data for new actions
new_sequences, new_labels = [], []
for action in actions:
    for sequence in range(1, no_sequences+1):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        new_sequences.append(window)
        new_labels.append(label_map[action])

# Convert new data to numpy arrays
X_new = np.array(new_sequences)
y_new = to_categorical(new_labels, num_classes=len(all_actions)).astype(int)

# Append new data to existing training data
if 'X_train' not in locals() or 'y_train' not in locals():
    X_train, y_train = X_new, y_new
else:
    X_train = np.concatenate((X_train, X_new))
    y_train = np.concatenate((y_train, y_new))

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, callbacks=[TensorBoard(log_dir='Logs')])

# Save the updated model
model.save('updated_action.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print("Test Accuracy:", accuracy)
