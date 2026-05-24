import gradio as gr
import librosa
import numpy as np
import joblib
from tensorflow import keras

model = keras.models.load_model('CNN-2D_accent_model_check_CHECK.keras')

def extract_features(audio_path, sr=22050, duration=3, n_mfcc=20, n_mels=128):
    try:
        # Load audio with fixed duration
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Pad if too short
        if len(audio) < sr * duration:
            audio = np.pad(audio, (0, sr * duration - len(audio)), mode='constant')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)
        harmonic = librosa.effects.harmonic(y=audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

        # Aligning all feature matrices to the same number of frames
        min_frames = min(mfcc.shape[1], mel_spec_db.shape[1], spec_contrast.shape[1],
                         chroma.shape[1], zcr.shape[1], rms.shape[1], tonnetz.shape[1])

        # Sliceing to the minimum length
        mfcc = mfcc[:, :min_frames]
        mel_spec_db = mel_spec_db[:, :min_frames]
        spec_contrast = spec_contrast[:, :min_frames]
        chroma = chroma[:, :min_frames]
        zcr = zcr[:, :min_frames]
        rms = rms[:, :min_frames]
        tonnetz = tonnetz[:, :min_frames]
        
        # Stack features vertically
        combined = np.vstack([
            mfcc,
            mel_spec_db,
            spec_contrast,
            chroma,
            zcr,
            rms,
            tonnetz
        ])

        return combined

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


def predict_accent(audio_file):
    """
    Predict accent for uploaded audio file using 2D CNN model
    """
    # Handle None input
    if audio_file is None:
        return "Please upload an audio file."
    
    # Extract features
    features = extract_features(audio_file)
    if features is None:
        return "Error extracting features. Please try with a different audio file."
    
    # Reshape for CNN input: (batch_size, height, width, channels)
    # Add channel dimension and batch dimension
    features = np.expand_dims(features, axis=-1)  # Add channel dimension -> (features, time, 1)
    features = np.expand_dims(features, axis=0)   # Add batch dimension -> (1, features, time, 1)
    
    # Make prediction (no scaler needed for CNN)
    prediction = model.predict(features, verbose=0)
    probability = prediction[0][0]
    
    # Interpret result
    if probability > 0.5:
        accent_label = "Tamil Nadu"
        confidence = probability * 100
    else:
        accent_label = "Kerala"
        confidence = (1 - probability) * 100
    
    # Format detailed output
    result = f"""
🎤 Accent Prediction Results --

Predicted Accent: {accent_label}
Confidence: {confidence:.2f}%

Probability Distribution:
- Kerala: {(1-probability)*100:.2f}%
- Tamil Nadu: {probability*100:.2f}%

Raw Probability Score: {probability:.4f}
    """
    
    return result


# Create Gradio interface
demo = gr.Interface(
    fn=predict_accent,
    inputs=gr.Audio(
        sources=["upload", "microphone"],  # Allow both file upload and recording
        type="filepath",
        label="Upload Audio File or Record"
    ),
    outputs=gr.Textbox(
        label="Accent Classification Result",
        lines=10
    ),
    title="🗣️ Kerala vs Tamil Nadu English Accent Classifier",
    description="""
    Upload an audio file or record your voice speaking English to classify the accent.
    This model uses a 2D CNN trained on audio features to distinguish between Kerala and Tamil Nadu English accents.
    
    **Supported formats:** WAV
    **Recommended:** Clear audio with minimal background noise
    """,
    examples=[
        # Add example audio files if you have them
        # ["example_kerala.wav"],
        # ["example_tamilnadu.wav"]
    ],
    theme="default",
    allow_flagging="never"
)

demo.launch(inline=True)