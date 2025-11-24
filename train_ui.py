import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Dense, Dropout, Flatten, 
    Activation, GlobalAveragePooling2D, Multiply, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.utils import to_categorical

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

PROCESSED_DATA_PATH = '/home/spectre/Projects/beta_dataset/dataset/Processed_BETA_UI'

# BETA Constants
N_CLASSES = 40
N_CHANNELS_ORIG = 64
N_CHANNELS_PAD = 66 # 64 + 2 padding
N_FEATURES = 476    # Based on preprocessing

# Training Settings (User Independent)
BATCH_SIZE = 32     
EPOCHS = 50         
LEARNING_RATE = 0.001

# Input Shape (Channels, Features, 1)
INPUT_SHAPE = (N_CHANNELS_PAD, N_FEATURES, 1)

# ==========================================
# 2. DEBUG & SETUP
# ==========================================

def check_path():
    print(f"--- CHECKING DATA PATH: {PROCESSED_DATA_PATH} ---")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("ERROR: The folder does not exist!")
        return False
    
    # Look for any .npy file
    files = os.listdir(PROCESSED_DATA_PATH)
    npy_files = [f for f in files if f.endswith('.npy')]
    
    if not npy_files:
        print(f"ERROR: Folder exists but contains NO .npy files.")
        return False
        
    print(f"✅ SUCCESS: Found {len(npy_files)} files.")
    return True

# ==========================================
# 3. MODEL DEFINITION (Adapted FBCNN)
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
    
    # Final Dense Layer (40 classes)
    out = Dense(units=N_CLASSES, activation='softmax',
                kernel_regularizer=l2(decay),
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(concat)
    
    model = Model(inputs=[input1, input2, input3], outputs=out)
    return model

# ==========================================
# 4. DATA GENERATOR (Prevents RAM Crash)
# ==========================================

def data_generator(subject_list):
    """Yields data batch-by-batch from disk."""
    for subject in subject_list:
        try:
            f1 = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_fb1.npy"))
            f2 = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_fb2.npy"))
            f3 = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_fb3.npy"))
            lb = np.load(os.path.join(PROCESSED_DATA_PATH, f"{subject}_labels.npy"))
            
            # Expand dims
            f1 = np.expand_dims(f1, axis=-1)
            f2 = np.expand_dims(f2, axis=-1)
            f3 = np.expand_dims(f3, axis=-1)
            
            for i in range(len(lb)):
                yield (f1[i], f2[i], f3[i]), lb[i]
                
        except Exception as e:
            print(f"Error loading {subject}: {e}")
            continue

def create_dataset(subject_list, is_training=True):
    output_sig = (
        (tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
         tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
         tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    ds = tf.data.Dataset.from_generator(lambda: data_generator(subject_list), output_signature=output_sig)
    
    # One-hot encode labels
    def process_item(inputs, label):
        return inputs, tf.one_hot(label, N_CLASSES)
    
    ds = ds.map(process_item, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.shuffle(1000)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# 5. MAIN TRAINING LOOP (User Independent - LOGO)
# ==========================================

if __name__ == '__main__':
    
    if not check_path():
        exit()
    
    # 1. Find all subjects
    label_files = glob.glob(os.path.join(PROCESSED_DATA_PATH, "*_labels.npy"))
    all_subjects = [os.path.basename(f).replace('_labels.npy', '') for f in label_files]
    
    # Filter for 'S' subjects
    all_subjects = [s for s in all_subjects if s.startswith('S')]
    all_subjects.sort()
    
    if not all_subjects:
        print("ERROR: No subjects found starting with 'S'. Check filenames.")
        exit()

    print(f"Found {len(all_subjects)} subjects. Starting UI (LOGO) Training...")

    fold_accuracies = []

    # List to store results for CSV
    fold_results = []

    # 2. Leave-One-Group-Out Loop
    # We iterate through each subject, setting them as the TEST subject once
    for i, test_subject in enumerate(all_subjects):
        print(f"\n" + "="*40)
        print(f"FOLD {i+1}/{len(all_subjects)}")
        print(f"TESTING ON: {test_subject}")
        print("="*40)
        
        # Split: Test on 1 subject, Train on everyone else
        train_subjects = [s for s in all_subjects if s != test_subject]
        
        # Create Data Generators
        train_ds = create_dataset(train_subjects, is_training=True)
        test_ds = create_dataset([test_subject], is_training=False)
        
        # Build Model
        sgd = optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=False)
        
        model = cnn_model(INPUT_SHAPE, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        # Train
        model.fit(train_ds, epochs=EPOCHS, verbose=1)
        
        # Evaluate
        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f"\n>>> Accuracy for {test_subject}: {acc*100:.2f}%")

        fold_accuracies.append(acc)

        # Save fold result
        fold_results.append({
            'fold': i + 1,
            'test_subject': test_subject,
            'num_train_subjects': len(train_subjects),
            'accuracy': acc,
            'loss': loss
        })

        # Running Average
        print(f"OVERALL AVERAGE SO FAR: {np.mean(fold_accuracies)*100:.2f}%")

        # Free model memory (TensorFlow will handle GPU cleanup)
        del model

    # Cleanup TensorFlow session after all folds complete
    tf.keras.backend.clear_session()

    # 3. Final Results
    print("\n" + "="*40)
    print(f"FINAL AVERAGE UI ACCURACY ({len(all_subjects)} subjects): {np.mean(fold_accuracies)*100:.2f}%")
    print(f"STD DEV: {np.std(fold_accuracies)*100:.2f}%")
    print("="*40)

    # 4. Save results to CSV
    # Save fold-level results
    fold_df = pd.DataFrame(fold_results)
    fold_csv = 'results_ui_folds.csv'
    fold_df.to_csv(fold_csv, index=False)
    print(f"\n✅ Fold results saved to: {fold_csv}")

    # Save overall summary
    summary_df = pd.DataFrame([{
        'total_folds': len(all_subjects),
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE
    }])
    summary_csv = 'results_ui_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary saved to: {summary_csv}")