import streamlit as st
import os
import tempfile
import logging
from moviepy import VideoFileClip
from faster_whisper import WhisperModel

# ==========================================
# BACKEND FUNCTIONS (From your pipeline)
# ==========================================

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio_from_video(video_filepath: str, output_audio_filepath: str) -> bool:
    """Extracts the audio track from a video file and saves it locally."""
    if not os.path.exists(video_filepath):
        logging.error(f"Input video file not found at: {video_filepath}")
        return False

    video_clip = None
    audio_clip = None

    try:
        logging.info(f"Loading video file: {video_filepath}")
        video_clip = VideoFileClip(video_filepath)
        
        if video_clip.audio is None:
            logging.error("No audio track found in the provided video file.")
            return False
            
        audio_clip = video_clip.audio
        logging.info(f"Extracting audio to: {output_audio_filepath}")
        
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
        if audio_clip is not None:
            audio_clip.close()
        if video_clip is not None:
            video_clip.close()

def transcribe_audio_locally(audio_filepath: str, output_text_filepath: str, model_size: str = "base") -> bool:
    """Transcribes an audio file to text using a locally running Whisper model."""
    if not os.path.exists(audio_filepath):
        logging.error(f"Audio file not found at: {audio_filepath}")
        return False

    try:
        logging.info(f"Loading local Whisper model ('{model_size}')...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        logging.info(f"Starting transcription for: {audio_filepath}")
        segments, info = model.transcribe(audio_filepath, beam_size=5)
        
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        with open(output_text_filepath, "w", encoding="utf-8") as file:
            file.write(f"--- Audio Transcript ---\n")
            file.write(f"Language: {info.language}\n\n")
            
            for segment in segments:
                logging.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                file.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}\n")
        
        logging.info(f"Transcription successfully saved to: {output_text_filepath}")
        return True

    except Exception as e:
        logging.error(f"An error occurred during transcription: {str(e)}")
        return False


# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="EchoFrame", page_icon="🎥", layout="wide")

# Initialize session state so the transcript survives page navigation
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None

st.sidebar.title("🎥 EchoFrame")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["1. Upload & Process", "2. View Transcript"])

# --- PAGE 1: UPLOAD & PROCESS ---
if page == "1. Upload & Process":
    st.title("Step 1: Secure Video Ingestion")
    st.write("Upload a client video securely. All audio extraction and transcription runs locally.")
    
    uploaded_file = st.file_uploader("Upload Presentation Video", type=['mp4', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Process Video", type="primary"):
            # Create a progress container to update the user
            status_text = st.empty()
            
            # Set up temporary file paths
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, "temp_input_video.mp4")
            temp_audio_path = os.path.join(temp_dir, "temp_extracted_audio.wav")
            temp_text_path = os.path.join(temp_dir, "temp_transcript.txt")
            
            try:
                # 1. Save uploaded video to disk
                status_text.info("Saving uploaded video to local storage...")
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # 2. Extract Audio
                status_text.info("Extracting audio stream using MoviePy (this may take a moment)...")
                extraction_success = extract_audio_from_video(temp_video_path, temp_audio_path)
                
                if not extraction_success:
                    st.error("Audio extraction failed. Please check the logs.")
                else:
                    # 3. Transcribe Audio
                    status_text.info("Transcribing audio via Faster-Whisper. Hang tight...")
                    transcription_success = transcribe_audio_locally(temp_audio_path, temp_text_path, model_size="base")
                    
                    if not transcription_success:
                        st.error("Transcription failed. Please check the logs.")
                    else:
                        # 4. Read the generated transcript file into Streamlit's state
                        with open(temp_text_path, "r", encoding="utf-8") as file:
                            st.session_state.transcript = file.read()
                        
                        st.session_state.video_name = uploaded_file.name
                        status_text.success("✅ Processing complete! Please navigate to the 'View Transcript' page.")
                        
            finally:
                # 5. Clean up temporary files securely
                for temp_file in [temp_video_path, temp_audio_path, temp_text_path]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logging.error(f"Could not remove temp file {temp_file}: {e}")

# --- PAGE 2: VIEW TRANSCRIPT ---
elif page == "2. View Transcript":
    st.title("Step 2: Transcript Review")
    
    if st.session_state.transcript is None:
        st.warning("No transcript found. Please go back to the 'Upload & Process' page and submit a video first.")
    else:
        st.success(f"Viewing transcript for: **{st.session_state.video_name}**")
        
        st.text_area("Extracted Text:", value=st.session_state.transcript, height=400)
        
        st.download_button(
            label="Download Transcript as .txt",
            data=st.session_state.transcript,
            file_name=f"{os.path.splitext(st.session_state.video_name)[0]}_transcript.txt",
            mime="text/plain"
        )