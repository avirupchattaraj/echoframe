import os
import logging
from faster_whisper import WhisperModel

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio_locally(audio_filepath: str, output_text_filepath: str, model_size: str = "base") -> bool:
    """
    Transcribes an audio file to text using a locally running Whisper model
    and saves the output to a standard text file.
    
    Args:
        audio_filepath (str): Path to the input audio file (e.g., extracted_speech.wav).
        output_text_filepath (str): Path to save the final transcript (e.g., transcript.txt).
        model_size (str): Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large-v3').
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(audio_filepath):
        logging.error(f"Audio file not found at: {audio_filepath}")
        return False

    try:
        logging.info(f"Loading local Whisper model ('{model_size}')...")
        # device="auto" uses GPU if available, else CPU.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        logging.info(f"Starting transcription for: {audio_filepath}")
        segments, info = model.transcribe(audio_filepath, beam_size=5)
        
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        # Open the file in write mode
        with open(output_text_filepath, "w", encoding="utf-8") as file:
            file.write(f"--- Audio Transcript: {os.path.basename(audio_filepath)} ---\n")
            file.write(f"Language: {info.language}\n\n")
            
            for segment in segments:
                # Log to console
                logging.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                
                # Write to the file, including the timestamps which will be useful for Phase 3
                file.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}\n")
        
        logging.info(f"Transcription successfully saved to: {output_text_filepath}")
        return True

    except Exception as e:
        logging.error(f"An error occurred during transcription: {str(e)}")
        return False

if __name__ == "__main__":
    input_audio = "extracted_speech.wav"
    output_transcript = "transcript.txt"  # This is your "normal file" output
    
    success = transcribe_audio_locally(input_audio, output_transcript, model_size="base")
    
    if success:
        print(f"\nYour transcript has been downloaded and saved to {output_transcript}.")