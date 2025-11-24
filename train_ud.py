import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Dense, Dropout, Flatten, 
    Activation, GlobalAveragePooling2D, Multiply, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

# ==========================================
# 0. GPU CONFIGURATION
# ==========================================

# Configure GPU to allow memory growth (prevents TensorFlow from allocating all GPU memory)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU ENABLED: Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  WARNING: No GPU found. Training will use CPU (much slower).")

# ==========================================
# 1. CONFIGURATION
# ==========================================

# PATH to the folder with .npy files (Same preprocessing folder as before)
PROCESSED_DATA_PATH = '/home/spectre/Projects/beta_dataset/dataset/Processed_BETA_UI'

# BETA Constants
N_CLASSES = 40
N_CHANNELS_ORIG = 64
N_CHANNELS_PAD = 66 # 64 + 2 padding
N_FEATURES = 476    # Based on preprocessing

# Training Settings (User Dependent)
# UD often uses smaller batch sizes because we have less data per fold
BATCH_SIZE = 16     
EPOCHS = 50         
LEARNING_RATE = 0.001
N_FOLDS = 10        # 10-fold cross-validation per subject

# Input Shape
INPUT_SHAPE = (N_CHANNELS_PAD, N_FEATURES, 1)

# ==========================================
# 2. MODEL DEFINITION (Same FBCNN Architecture)
# ==========================================

def cnn_model(input_shape, decay=0.001):
    # Define 3 Inputs
    input1 = Input(shape=input_shape, name='input_1')
    input2 = Input(shape=input_shape, name='input_2')
    input3 = Input(shape=input_shape, name='input_3')
    
    # Shared Subnet
    def subnet_layers(x_in):
        # Layer 1: Time/Freq Conv
        x = Conv2D(filters=16, kernel_size=(3,3), use_bias=False, padding='valid',
                   kernel_regularizer=l2(decay),
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(x_in)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        
        # Layer 2: Spatial Conv (64 channels)
        x = Conv2D(filters=32, kernel_size=(N_CHANNELS_ORIG, 1), use_bias=False, padding='valid',
                   kernel_regularizer=l2(decay),
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        
        # Layer 3: Feature Conv
        x = Conv2D(filters=64, kernel_size=(1, 10), use_bias=False, padding='valid',
                   kernel_regularizer=l2(decay),
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        
        x = Flatten()(x)
        return x

    # Apply subnets
    x1 = subnet_layers(input1)
    x2 = subnet_layers(input2)
    x3 = subnet_layers(input3)
    
    # Output Logic
    add1 = Add()([x1, x2])
    add2 = Add()([x1, x3])
    add3 = Add()([x2, x3])
    
    concat = Concatenate()([x1, x2, x3, add1, add2, add3])
    
    out = Dense(units=N_CLASSES, activation='softmax',
                kernel_regularizer=l2(decay),
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(concat)
    
    model = Model(inputs=[input1, input2, input3], outputs=out)
    return model

# ==========================================
# 3. MAIN TRAINING LOOP (User Dependent)
# ==========================================

if __name__ == '__main__':
    
    # 1. Find all subjects
    label_files = glob.glob(os.path.join(PROCESSED_DATA_PATH, "*_labels.npy"))
    if not label_files:
        print(f"ERROR: No data found in {PROCESSED_DATA_PATH}")
        exit()
        
    all_subjects = [os.path.basename(f).replace('_labels.npy', '') for f in label_files]
    all_subjects.sort()
    
    print(f"Found {len(all_subjects)} subjects. Starting UD Training...")
    
    final_subject_accuracies = []

    # 2. Loop through each subject individually
    for sub_idx, subject in enumerate(all_subjects):
        print(f"\n" + "="*40)
        print(f"SUBJECT {sub_idx+1}/{len(all_subjects)}: {subject}")
        print("="*40)
        
        # Load THIS subject's data into memory
        # (For UD, data is small enough to fit in RAM: 160 trials)
        try:
            f1 = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_fb1.npy"))
            f2 = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_fb2.npy"))
            f3 = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_fb3.npy"))
            labels = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_labels.npy"))
            
            # Expand dims: (Trials, 66, 476) -> (Trials, 66, 476, 1)
            f1 = np.expand_dims(f1, axis=-1)
            f2 = np.expand_dims(f2, axis=-1)
            f3 = np.expand_dims(f3, axis=-1)
            
            # One-hot encode labels
            labels_onehot = to_categorical(labels, num_classes=N_CLASSES)
            
        except Exception as e:
            print(f"Error loading {subject}: {e}")
            continue

        # 3. 10-Fold Cross Validation for THIS subject
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_accs = []

        # Build model once and reuse across all folds (just reset weights)
        sgd = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=False)
        model = cnn_model(INPUT_SHAPE, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # Save initial weights for resetting between folds
        initial_weights = model.get_weights()

        for fold, (train_idx, test_idx) in enumerate(kf.split(f1)):
            print(f"  Fold {fold+1}/{N_FOLDS}...", end="")

            # Reset model weights to fresh random values for this fold
            if fold > 0:
                model.set_weights(initial_weights)

            # Split data
            X_train = [f1[train_idx], f2[train_idx], f3[train_idx]]
            y_train = labels_onehot[train_idx]

            X_test = [f1[test_idx], f2[test_idx], f3[test_idx]]
            y_test = labels_onehot[test_idx]

            # Train
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            # Evaluate
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            fold_accs.append(acc)
            print(f" Acc: {acc*100:.2f}%")

        # Free model after all folds complete
        del model

        # Average accuracy for this subject
        avg_acc = np.mean(fold_accs)
        final_subject_accuracies.append(avg_acc)
        print(f"  >> Average Accuracy for {subject}: {avg_acc*100:.2f}%")

        # Cleanup TensorFlow session after all folds for this subject
        tf.keras.backend.clear_session()

    # 4. Final Results
    print("\n" + "="*40)
    print(f"FINAL AVERAGE UD ACCURACY ({len(all_subjects)} subjects): {np.mean(final_subject_accuracies)*100:.2f}%")
    print(f"STD DEV: {np.std(final_subject_accuracies)*100:.2f}%")
    print("="*40)