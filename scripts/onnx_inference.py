import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import math
from pathlib import Path

# Assuming 'demucs-for-onnx' is installed or accessible in the Python path
# If not, adjust sys.path or install the package
try:
    from demucs.htdemucs import standalone_spec, standalone_magnitude, standalone_mask, standalone_ispec
    from demucs.spec import spectro, ispectro
    from demucs.hdemucs import pad1d
    from demucs.utils import center_trim
except ImportError as e:
    print(f"Error importing demucs functions: {e}")
    print("Please ensure 'demucs-for-onnx' is installed (e.g., pip install -e ./demucs-for-onnx)")
    print("Or adjust the Python path if it's located elsewhere.")
    exit(1)

# --- Constants ---
ONNX_MODEL_PATH = Path('./model_output/v1_stft/htdemucs_ft.onnx')
INPUT_AUDIO_PATH = Path("charli.flac") # Replace with your desired input audio file
OUTPUT_DIR = Path("audio_output/onnx_inference_test/")

TARGET_LENGTH = 343980  # Exact input length required by the user
SAMPLERATE = 44100

NFFT = 4096
HOP_LENGTH = NFFT // 4 # 1024

# Assumed ONNX input/output names (based on model structure and conversion script analysis)
ONNX_OUTPUT_NAMES = ['out_x', 'out_xt']

# --- Main Script ---
def main():
    if not ONNX_MODEL_PATH.exists():
        print(f"Error: ONNX model not found at {ONNX_MODEL_PATH}")
        return

    if not INPUT_AUDIO_PATH.exists():
        print(f"Error: Input audio not found at {INPUT_AUDIO_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Audio
    print(f"Loading audio: {INPUT_AUDIO_PATH}")
    try:
        input_audio, sr = sf.read(INPUT_AUDIO_PATH, dtype='float32')
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # 2. Prepare Audio Tensor
    print(f"Original audio shape: {input_audio.shape}, Samplerate: {sr}")
    if sr != SAMPLERATE:
        print(f"Warning: Audio samplerate ({sr}) doesn't match model ({SAMPLERATE}). Resampling is needed (not implemented).")
        # Add resampling logic here if needed (e.g., using librosa or torchaudio)
        return

    # Convert to stereo if mono
    if input_audio.ndim == 1:
        input_audio = np.stack([input_audio, input_audio], axis=-1)
    elif input_audio.shape[-1] != 2:
        print(f"Error: Unsupported number of channels ({input_audio.shape[-1]}). Expected mono or stereo.")
        return

    # Transpose to (channels, samples) and add batch dimension -> [1, 2, samples]
    input_tensor = torch.from_numpy(input_audio.T).unsqueeze(0).float()
    print(f"Input tensor shape after transpose/unsqueeze: {input_tensor.shape}")

    # 3. Trim/Pad to TARGET_LENGTH
    current_length = input_tensor.shape[-1]
    if current_length > TARGET_LENGTH:
        print(f"Trimming audio from {current_length} to {TARGET_LENGTH}")
        input_snippet = center_trim(input_tensor, TARGET_LENGTH)
    elif current_length < TARGET_LENGTH:
        print(f"Padding audio from {current_length} to {TARGET_LENGTH}")
        padding_needed = TARGET_LENGTH - current_length
        input_snippet = F.pad(input_tensor, (0, padding_needed)) # Pad right
    else:
        input_snippet = input_tensor
    print(f"Input snippet shape (trimmed/padded): {input_snippet.shape}")


    # 6. ONNX Inference
    print(f"Loading ONNX model: {ONNX_MODEL_PATH}")
    try:
        session = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH), providers=['CPUExecutionProvider']) # Or CUDAExecutionProvider
        print("ONNX session created.")

        print("Running ONNX inference...")
        onnx_inputs = {
           "input_audio" : input_snippet.numpy(), # Shape [B, C, num_samples]
        }
        onnx_outputs = session.run(["outx", "outxt"], onnx_inputs)
        onnx_out_x = onnx_outputs[0]
        onnx_out_xt = onnx_outputs[1]
        print("ONNX inference complete.")
        print(f"  onnx_out_x shape: {onnx_out_x.shape}, dtype: {onnx_out_x.dtype}")
        print(f"  onnx_out_xt shape: {onnx_out_xt.shape}, dtype: {onnx_out_xt.dtype}")

    except Exception as e:
        print(f"Error during ONNX loading or inference: {e}")
        # Potentially print input/output names if error is related to them
        # meta = session.get_modelmeta()
        # print("Expected inputs:", [inp.name for inp in session.get_inputs()])
        # print("Expected outputs:", [out.name for out in session.get_outputs()])
        return

    # 7. Postprocessing
    print("Postprocessing outputs...")
    # Convert outputs back to PyTorch tensors
    out_x = torch.from_numpy(onnx_out_x)
    out_xt = torch.from_numpy(onnx_out_xt)

    # Apply mask (using the PyTorch magspec tensor)

    magspec = standalone_magnitude(standalone_spec(input_snippet, nfft=NFFT, hop_length=HOP_LENGTH))
    print("Applying mask...")
    zout = standalone_mask(magspec, out_x) # magspec is [B, C*2, F, T], out_x is [B, S, C*2, F, T]
    print(f"Masked complex spectrogram (zout) shape: {zout.shape}") # Should be [B, S, C, F, T] complex

    # Inverse STFT
    print("Performing inverse STFT...")
    # standalone_ispec expects complex [B, S, C, F, T]
    final_out_component = standalone_ispec(zout, TARGET_LENGTH)
    print(f"Component after iSTFT shape: {final_out_component.shape}") # Should be [B, S, C, T_train]

    # Combine with time-domain output
    print("Combining time and frequency components...")
    # Ensure shapes match before adding: out_xt should be [B, S, C, T_train]
    if out_xt.shape != final_out_component.shape:
         print(f"Warning: Shape mismatch between out_xt {out_xt.shape} and final_out_component {final_out_component.shape}. Check ONNX output shapes.")
         # Attempt to trim/pad out_xt if necessary, assuming time dimension mismatch
         diff = out_xt.shape[-1] - final_out_component.shape[-1]
         if diff > 0:
             out_xt = out_xt[..., :final_out_component.shape[-1]]
         elif diff < 0:
             out_xt = F.pad(out_xt, (0, -diff))
         print(f"Adjusted out_xt shape: {out_xt.shape}")


    final_combined_out = out_xt + final_out_component
    print(f"Combined output shape: {final_combined_out.shape}")

    # 8. Final Trimming
    print(f"Trimming final output to target length {TARGET_LENGTH}...")
    final_trimmed = center_trim(final_combined_out, TARGET_LENGTH)
    print(f"Final trimmed output shape: {final_trimmed.shape}, dtype: {final_trimmed.dtype}") # Should be [B, S, C, T_target]

    # 9. Save Output (Optional)
    # Detach tensor, move to CPU, convert to numpy, transpose back to [samples, channels]
    final_trimmed_np = final_trimmed.squeeze(0).detach().cpu().numpy() # Shape [S, C, T_target]
    num_sources = final_trimmed_np.shape[0]
    sources_list = ["drums", "bass", "other", "vocals"] # Assuming 4s model, adjust if needed

    print(f"Saving {num_sources} output sources to {OUTPUT_DIR}...")
    for i in range(num_sources):
        source_audio = final_trimmed_np[i].T # Shape [T_target, C]
        source_name = sources_list[i] if i < len(sources_list) else f"source_{i}"
        output_path = OUTPUT_DIR / f"{INPUT_AUDIO_PATH.stem}_{source_name}.wav"
        try:
            sf.write(str(output_path), source_audio, SAMPLERATE)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error saving {output_path}: {e}")

    print("Script finished.")

if __name__ == "__main__":
    main()
