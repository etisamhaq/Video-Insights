import os
import subprocess
import streamlit as st
import tempfile
import whisper
import json
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import ffmpeg

# Set page configuration
st.set_page_config(
    page_title="Video Processing System",
    page_icon="🎬",
    layout="wide"
)

# # Set up FFmpeg path (to fix FileNotFoundError)
# FFMPEG_PATH = "/app/.bin/ffmpeg"

# if not os.path.exists(FFMPEG_PATH):
#     st.warning("Downloading FFmpeg, please wait...")
#     subprocess.run("mkdir -p /app/.bin && wget -q -O /app/.bin/ffmpeg https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz && tar -xf /app/.bin/ffmpeg-release-i686-static.tar.xz -C /app/.bin --strip-components=1", shell=True)
#     subprocess.run("chmod +x /app/.bin/ffmpeg", shell=True)

# # Add FFmpeg binary to system PATH
# os.environ["PATH"] += os.pathsep + "/app/.bin"

# # Function to extract audio from video using FFmpeg
# def extract_audio(video_path, output_audio_path):
#     command = f'{FFMPEG_PATH} -i "{video_path}" -q:a 0 -map a "{output_audio_path}" -y'
#     subprocess.call(command, shell=True)
#     return output_audio_path


# Preconfigured settings
WHISPER_MODEL_SIZE = "tiny"  # Using tiny model as requested
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Set up the Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model(WHISPER_MODEL_SIZE)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, whisper_model):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Function to extract audio from video

def extract_audio(video_path, output_audio_path):
    try:
        ffmpeg.input(video_path).output(output_audio_path, format="mp3", acodec="libmp3lame").run(overwrite_output=True)
        return output_audio_path
    except ffmpeg.Error as e:
        st.error(f"FFmpeg error: {e}")
        return None

# Function to process LLM requests using Groq
def process_with_llm(text, prompt, temperature=0.7):
    # Initialize ChatGroq with preconfigured API key
    chat = ChatGroq(
        model_name="llama3-70b-8192",  # Using Llama3 70B model from Groq
        temperature=temperature,
        groq_api_key=GROQ_API_KEY
    )
    
    # Create messages
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text)
    ]
    
    # Get response
    response = chat(messages)
    return response.content

# Initialize session state variables
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Main app UI
st.title("🎬 Frame Wise - Video Processing System")

# Sidebar for app information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application allows you to:
    1. Upload a video or provide a local path
    2. Transcribe the audio using Whisper (tiny model)
    3. Process the transcription with Groq LLM for:
       - Summarization
       - Key points extraction
       - Interactive chat
    """)
    
    st.markdown("---")
    st.markdown(f"**Model Configuration:**")
    st.markdown(f"- Whisper: {WHISPER_MODEL_SIZE}")
    st.markdown(f"- LLM: Groq (llama3-70b-8192)")

# Main content area
tab1, tab2, tab3 = st.tabs(["Upload & Transcribe", "Analysis", "Chat"])

with tab1:
    st.header("Step 1: Upload Video or Provide Path")
    
    source_option = st.radio("Select video source:", ["Upload Video", "Local Path"])
    
    if source_option == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
    else:
        video_path = st.text_input("Enter the path to your local video file:")
    
    st.header("Step 2: Transcribe Video")
    
    if st.button("Start Transcription") and (('uploaded_file' in locals() and uploaded_file is not None) or ('video_path' in locals() and video_path)):
        with st.spinner("Extracting audio from video..."):
            # Create a temporary file for the audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            extract_audio(video_path, temp_audio)
        
        with st.spinner(f"Transcribing audio with Whisper ({WHISPER_MODEL_SIZE})..."):
            # Load Whisper model
            whisper_model = load_whisper_model()
            
            # Transcribe the audio
            st.session_state.transcription = transcribe_audio(temp_audio, whisper_model)
            st.session_state.processing_done = True
            
            # Clean up temp files
            os.remove(temp_audio)
            if source_option == "Upload Video":
                os.remove(video_path)
        
        st.success("Transcription complete!")
    
    if st.session_state.transcription:
        st.header("Transcription Result")
        st.text_area("Transcribed Text", st.session_state.transcription, height=300)

with tab2:
    st.header("Video Analysis")
    
    if not st.session_state.processing_done:
        st.info("Please complete the transcription in the 'Upload & Transcribe' tab first.")
    else:
        analysis_type = st.selectbox("Select Analysis Type", ["Summary", "Key Points", "Full Analysis"])
        
        if st.button("Generate Analysis"):
            with st.spinner("Processing with Groq LLM..."):
                if analysis_type == "Summary":
                    prompt = "You are an assistant that summarizes video transcripts. Create a concise summary of the following transcript:"
                    result = process_with_llm(st.session_state.transcription, prompt)
                    st.subheader("Summary")
                    st.write(result)
                
                elif analysis_type == "Key Points":
                    prompt = "You are an assistant that extracts key points from video transcripts. List the main points from the following transcript:"
                    result = process_with_llm(st.session_state.transcription, prompt)
                    st.subheader("Key Points")
                    st.write(result)
                
                elif analysis_type == "Full Analysis":
                    prompt = """You are an assistant that performs comprehensive analysis of video transcripts. 
                    Provide the following for the transcript:
                    1. A concise summary
                    2. Key points and takeaways
                    3. Main topics discussed
                    4. Any important quotes or statements
                    """
                    result = process_with_llm(st.session_state.transcription, prompt)
                    st.subheader("Full Analysis")
                    st.write(result)

with tab3:
    st.header("Chat with Video Content")
    
    if not st.session_state.processing_done:
        st.info("Please complete the transcription in the 'Upload & Transcribe' tab first.")
    else:
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about the video...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response
            with st.spinner("Thinking..."):
                prompt = f"""You are an assistant that answers questions based on video transcripts. 
                Use only the information provided in the transcript to answer the user's question.
                If the answer cannot be determined from the transcript, say so.
                
                Transcript: {st.session_state.transcription}
                """
                
                response = process_with_llm(user_input, prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response)

# Footer
st.markdown("---")
st.markdown("Video Processing System | Powered by Whisper and Groq LLM")
