import os
import logging
from moviepy import VideoFileClip

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio_from_video(video_filepath: str, output_audio_filepath: str) -> bool:
    """
    Extracts the audio track from a video file and saves it locally.
    
    Args:
        video_filepath (str): Path to the input video file (e.g., client_vid.mp4).
        output_audio_filepath (str): Path where the extracted audio will be saved (e.g., output.wav).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(video_filepath):
        logging.error(f"Input video file not found at: {video_filepath}")
        return False

    video_clip = None
    audio_clip = None

    try:
        logging.info(f"Loading video file: {video_filepath}")
        video_clip = VideoFileClip(video_filepath)
        
        # Check if the video actually contains an audio track
        if video_clip.audio is None:
            logging.error("No audio track found in the provided video file.")
            return False
            
        audio_clip = video_clip.audio
        
        logging.info(f"Extracting audio to: {output_audio_filepath}")
        
        # REMOVED 'verbose=False' for MoviePy 2.0 compatibility
        # logger=None handles the suppression of the progress bar
        audio_clip.write_audiofile(
            output_audio_filepath, 
            fps=16000, 
            nbytes=2, 
            buffersize=2000, 
            codec='pcm_s16le',
            logger=None 
        )
        
        logging.info("Audio extraction completed successfully.")
        return True

    except Exception as e:
        logging.error(f"An error occurred during extraction: {str(e)}")
        return False
        
    finally:
        # Crucial for memory management, especially with large client videos
        if audio_clip is not None:
            audio_clip.close()
        if video_clip is not None:
            video_clip.close()

if __name__ == "__main__":
    # Example usage
    input_video = "sample_presentation.mp4"
    output_audio = "extracted_speech.wav"
    
    success = extract_audio_from_video(input_video, output_audio)
    
    if success:
        print(f"Ready to pass {output_audio} to the local Whisper model!")