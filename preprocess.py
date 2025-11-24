import os
import glob
import numpy as np
import scipy.io as sio
from scipy import signal
import cupy as cp
from cupyx.scipy import signal as cp_signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_PATH = '/home/spectre/Projects/beta_dataset/dataset'
OUTPUT_FOLDER = 'Processed_BETA_UI'

# BETA Dataset Constants
FS = 250
WINDOW_SECONDS = 1
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
NUM_CHANNELS = 64
NUM_CLASSES = 40
NUM_BLOCKS = 4

START_SAMPLE = 140
END_SAMPLE = START_SAMPLE + WINDOW_SAMPLES

NYQUIST = FS / 2.0
BANDS = [
    (6.0, 16.0),
    (16.0, 32.0),
    (32.0, 64.0)
]

# ==========================================
# 2. GPU-ACCELERATED FUNCTIONS
# ==========================================

def get_butter_filter(lowcut, highcut, order=4):
    """Creates filter coefficients (CPU operation, done once)."""
    low = lowcut / NYQUIST
    high = highcut / NYQUIST
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a

def process_one_subject_gpu(file_path, output_dir, filters_cpu):
    """
    GPU-accelerated version: Processes entire subject data on GPU.
    """
    subject_name = os.path.basename(file_path).replace('.mat', '')
    print(f"--- Processing {subject_name} (GPU) ---")

    try:
        # 1. Load Data (CPU)
        mat_contents = sio.loadmat(file_path)
        raw_eeg = mat_contents['data'][0, 0]['EEG']  # (64, 1000, 4, 40)
        
        # 2. Reshape for batch processing: (Classes*Blocks, Channels, Time)
        # Reorder to (Classes, Blocks, Channels, Time) then reshape
        raw_eeg = np.transpose(raw_eeg, (3, 2, 0, 1))  # (40, 4, 64, 1000)
        raw_eeg = raw_eeg.reshape(NUM_CLASSES * NUM_BLOCKS, NUM_CHANNELS, -1)  # (160, 64, 1000)
        
        # 3. Crop all trials at once
        trials_cropped = raw_eeg[:, :, START_SAMPLE:END_SAMPLE]  # (160, 64, 250)
        
        # 4. Move to GPU
        trials_gpu = cp.asarray(trials_cropped, dtype=cp.float32)
        
        # 5. Transfer filters to GPU
        filters_gpu = [(cp.asarray(b), cp.asarray(a)) for b, a in filters_cpu]
        
        # 6. Process all three filter banks
        features_fb = []
        
        for b_gpu, a_gpu in filters_gpu:
            # Apply filter to ALL trials and channels at once
            # filtfilt expects 1D or 2D, so we reshape
            n_trials, n_ch, n_samples = trials_gpu.shape
            data_flat = trials_gpu.reshape(n_trials * n_ch, n_samples)
            
            # GPU filtering (batch operation)
            filtered_flat = cp_signal.filtfilt(b_gpu, a_gpu, data_flat, axis=-1)
            filtered = filtered_flat.reshape(n_trials, n_ch, n_samples)
            
            # Apply FFT to all at once
            fft_result = cp.fft.rfft(filtered, n=1024, axis=-1) / WINDOW_SAMPLES
            
            # Extract frequency range (24 to 262 bins)
            idx_start = 24
            idx_end = 262
            fft_slice = fft_result[:, :, idx_start:idx_end]
            
            # Concatenate real and imaginary parts
            real_part = cp.real(fft_slice)
            imag_part = cp.imag(fft_slice)
            features = cp.concatenate((real_part, imag_part), axis=-1)  # (160, 64, 476)
            
            features_fb.append(features)
        
        # 7. Move results back to CPU
        X1 = cp.asnumpy(features_fb[0])
        X2 = cp.asnumpy(features_fb[1])
        X3 = cp.asnumpy(features_fb[2])
        
        # 8. Channel padding (64 -> 66)
        X1 = np.concatenate((X1, X1[:, 0:2, :]), axis=1)
        X2 = np.concatenate((X2, X2[:, 0:2, :]), axis=1)
        X3 = np.concatenate((X3, X3[:, 0:2, :]), axis=1)
        
        # 9. Create labels
        y = np.repeat(np.arange(NUM_CLASSES), NUM_BLOCKS).astype(np.int32)
        
        # 10. Save to disk
        np.save(os.path.join(output_dir, f"{subject_name}_fb1.npy"), X1.astype(np.float32))
        np.save(os.path.join(output_dir, f"{subject_name}_fb2.npy"), X2.astype(np.float32))
        np.save(os.path.join(output_dir, f"{subject_name}_fb3.npy"), X3.astype(np.float32))
        np.save(os.path.join(output_dir, f"{subject_name}_labels.npy"), y)
        
        print(f"  -> Saved: {X1.shape}")
        return True
        
    except Exception as e:
        print(f"!!! Error processing {subject_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_one_subject_wrapper(args):
    """Wrapper for multiprocessing."""
    file_path, output_dir, filters = args
    return process_one_subject_gpu(file_path, output_dir, filters)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    
    # Check CuPy availability
    try:
        cp.cuda.Device(0).compute_capability
        print("✓ GPU detected and CuPy available!")
    except:
        print("WARNING: No GPU detected. Install CuPy with: pip install cupy-cuda11x")
        print("Falling back to CPU version would be recommended.")
        exit(1)
    
    # 1. Setup Output Directory
    save_dir = os.path.join(DATASET_PATH, OUTPUT_FOLDER)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output folder: {save_dir}")
    else:
        print(f"Saving to existing folder: {save_dir}")
    
    # 2. Pre-compute filters (CPU, done once)
    print("Computing filter coefficients...")
    filters = [get_butter_filter(l, h) for l, h in BANDS]
    
    # 3. Find all S*.mat files
    search_pattern = os.path.join(DATASET_PATH, 'S*.mat')
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"ERROR: No files found at {search_pattern}")
        exit(1)
    
    print(f"\nFound {len(files)} subjects.")
    print(f"Processing with GPU acceleration...\n")
    
    # 4. Process files
    # Option A: Sequential (one at a time on GPU)
    import time
    start_time = time.time()
    
    for f in files:
        process_one_subject_gpu(f, save_dir, filters)
    
    elapsed = time.time() - start_time
    print(f"\n✓ All Processing Complete in {elapsed:.2f} seconds!")
    print(f"  Average: {elapsed/len(files):.2f} sec/subject")
