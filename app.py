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
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1
if 'levels_completed' not in st.session_state:
    st.session_state.levels_completed = []
if 'level_scores' not in st.session_state:
    st.session_state.level_scores = {}


# Main app UI
st.title("🎬 Frame Wise - Video Processing System")

# Sidebar for app information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application allows you to:
    1. Upload a video or provide a local path
    2. Transcribe the audio using Whisper
    3. Process the transcription with LLM for:
       - Summarization
       - Key points extraction
       - Interactive chat
       - Assessment with multiple choice questions (MCQs)
       - Generate study material based on incorrect answers
       - Track your progress through different levels
                
    """)
    

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Analysis", "Chat", "Assessment"])

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

with tab4:
    st.header("Assessment")
    
    if not st.session_state.processing_done:
        st.info("Please complete the transcription in the 'Upload & Transcribe' tab first.")
    else:
        # Show overall progress
        st.progress(len(st.session_state.levels_completed) / 4)
        
        # Initialize MCQ states if not exists
        if 'mcqs_generated' not in st.session_state:
            st.session_state.mcqs_generated = False
        if 'mcqs' not in st.session_state:
            st.session_state.mcqs = None
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = {}
            
        # Create level tabs
        level_tabs = st.tabs([f"Level {i}" for i in range(1, 5)])
        
        # Add new function to analyze mistakes and generate study material
        def analyze_mistakes_and_generate_material(mcqs, user_answers, level_idx):
            wrong_answers = []
            topics_to_review = set()
            
            for i, mcq in enumerate(mcqs):
                user_answer_index = mcq['options'].index(user_answers[i])
                if user_answer_index != mcq['correct_answer']:
                    wrong_answers.append({
                        'question': mcq['question'],
                        'user_answer': mcq['options'][user_answer_index],
                        'correct_answer': mcq['options'][mcq['correct_answer']],
                        'explanation': mcq['explanation']
                    })
                    topics_to_review.add(mcq['explanation'].split()[0])
            
            # Generate study material using LLM
            if topics_to_review:
                study_prompt = f"""Based on the user's incorrect answers in Level {level_idx}, create a comprehensive study guide that:
                1. Explains the relevant concepts clearly and concisely
                2. Provides practical examples
                3. Highlights key points to remember
                4. Includes 2-3 practice questions
                
                Focus on making the content easy to understand and remember.
                """
                
                study_material = process_with_llm(st.session_state.transcription, study_prompt, temperature=0.3, context=None)
                
                return {
                    'wrong_answers': wrong_answers,
                    'study_material': study_material
                }
            
            return None

        # Handle each level tab
        for level_idx, level_tab in enumerate(level_tabs, 1):
            with level_tab:
                if level_idx < st.session_state.current_level:
                    st.success(f"Level {level_idx} Completed! ✅")
                elif level_idx > st.session_state.current_level:
                    st.warning("🔒 Complete previous levels first")
                else:  # Current level
                    st.info(f"Current Level: {level_idx}")
                    
                    # Initialize level-specific states if not exists
                    if f'mcqs_level_{level_idx}' not in st.session_state:
                        st.session_state[f'mcqs_level_{level_idx}'] = None
                    if f'mcqs_generated_level_{level_idx}' not in st.session_state:
                        st.session_state[f'mcqs_generated_level_{level_idx}'] = False
                    if f'user_answers_level_{level_idx}' not in st.session_state:
                        st.session_state[f'user_answers_level_{level_idx}'] = {}
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        generate_button = st.button(
                            "Generate MCQs" if not st.session_state[f'mcqs_generated_level_{level_idx}'] else "Generate New MCQs",
                            key=f"gen_btn_{level_idx}"
                        )
                    with col2:
                        if st.session_state[f'mcqs_generated_level_{level_idx}']:
                            retake_button = st.button("Retake Test", key=f"retake_btn_{level_idx}")
                            if retake_button:
                                st.session_state[f'mcqs_generated_level_{level_idx}'] = False
                                st.session_state[f'mcqs_level_{level_idx}'] = None
                                st.session_state[f'user_answers_level_{level_idx}'] = {}
                                st.rerun()

                    # Generate MCQs for current level
                    if generate_button or (level_idx == st.session_state.current_level and not st.session_state[f'mcqs_generated_level_{level_idx}']):
                        with st.spinner(f"Generating MCQs for Level {level_idx}..."):
                            prompt = f"""You are an expert at creating multiple choice questions. 
                            This is for Level {level_idx} (questions should be {'more challenging than previous level' if level_idx > 1 else 'basic level'}).
                            Based on the following transcript, create exactly 10 MCQs. 
                            Your response must be a valid JSON array string in the following format:
                            [
                                {{
                                    "question": "What is...",
                                    "options": ["option1", "option2", "option3", "option4"],
                                    "correct_answer": 0,
                                    "explanation": "Brief explanation here"
                                }},
                                ...
                            ]
                            Requirements:
                            - Generate exactly 10 questions
                            - Make questions progressively harder for higher levels
                            - Each question must have exactly 4 options
                            - correct_answer must be 0, 1, 2, or 3
                            - Keep explanations concise
                            - Ensure the response is valid JSON
                            """
                            
                            try:
                                response = process_with_llm(st.session_state.transcription, prompt, temperature=0.7)
                                response = response.strip()
                                if not response.startswith('['):
                                    start = response.find('[')
                                    end = response.rfind(']') + 1
                                    if start != -1 and end != 0:
                                        response = response[start:end]
                                    else:
                                        raise json.JSONDecodeError("Invalid JSON format", response, 0)
                                
                                st.session_state[f'mcqs_level_{level_idx}'] = json.loads(response)
                                st.session_state[f'mcqs_generated_level_{level_idx}'] = True
                                st.success("MCQs generated successfully!")
                                
                            except Exception as e:
                                st.error(f"Error generating MCQs: {str(e)}")

                    if st.session_state[f'mcqs_generated_level_{level_idx}'] and st.session_state[f'mcqs_level_{level_idx}']:
                        st.subheader("Multiple Choice Questions")
                        
                        # Display MCQs with radio buttons
                        for i, mcq in enumerate(st.session_state[f'mcqs_level_{level_idx}']):
                            st.write(f"\n**Question {i+1}:** {mcq['question']}")
                            st.session_state[f'user_answers_level_{level_idx}'][i] = st.radio(
                                "Select your answer:",
                                mcq['options'],
                                key=f"mcq_{level_idx}_{i}"
                            )
                        
                        # Submit button for evaluation
                        if st.button("Submit Assessment", key=f"submit_{level_idx}"):
                            score = 0
                            wrong_answers = []
                            st.write("\n### Results")
                            
                            # First, collect all results
                            for i, mcq in enumerate(st.session_state[f'mcqs_level_{level_idx}']):
                                user_answer_index = mcq['options'].index(st.session_state[f'user_answers_level_{level_idx}'][i])
                                is_correct = user_answer_index == mcq['correct_answer']
                                
                                if is_correct:
                                    score += 1
                                else:
                                    wrong_answers.append({
                                        'question_num': i + 1,
                                        'question': mcq['question'],
                                        'user_answer': mcq['options'][user_answer_index],
                                        'correct_answer': mcq['options'][mcq['correct_answer']],
                                        'explanation': mcq['explanation']
                                    })
                            
                            percentage = (score / 10) * 100
                            st.write(f"\n## Final Score: {score}/10")
                            st.write(f"Percentage: {percentage}%")
                            
                            if percentage >= 50:
                                st.success(f"🎉 Congratulations! You've passed Level {level_idx}!")
                                
                                # Store the score for this level
                                st.session_state.level_scores[level_idx] = {
                                    'score': score,
                                    'percentage': percentage
                                }
                                
                                if level_idx not in st.session_state.levels_completed:
                                    st.session_state.levels_completed.append(level_idx)
                                
                                if level_idx < 4:
                                    st.session_state.current_level = level_idx + 1
                                    st.rerun()
                                else:
                                    st.balloons()
                                    st.success("🎓 Congratulations! You've completed all 4 levels!")
                            else:
                                st.warning("📚 Score below 50%")
                                st.error("Please review the study material and try again.")
                                
                                # Generate study material
                                if wrong_answers:
                                    # Display results for incorrect answers
                                    st.write("\n### Incorrect Answers")
                                    for wrong in wrong_answers:
                                        with st.expander(f"Question {wrong['question_num']}"):
                                            st.write(f"**Question:** {wrong['question']}")
                                            st.write(f"**Your Answer:** {wrong['user_answer']}")
                                            st.write(f"**Correct Answer:** {wrong['correct_answer']}")
                                            st.write(f"**Explanation:** {wrong['explanation']}")
                                
                                    # Generate study material using LLM
                                    study_prompt = f"""Based on the user's incorrect answers in Level {level_idx}, create a comprehensive study guide that:
                                    1. Explains the relevant concepts clearly and concisely
                                    2. Provides practical examples
                                    3. Highlights key points to remember
                                    4. Includes 2-3 practice questions
                                    
                                    Focus on making the content easy to understand and remember.
                                    """
                                    
                                    study_material = process_with_llm(st.session_state.transcription, study_prompt, temperature=0.3, context=None)
                                    
                                    # Display study material
                                    st.subheader("📚 Study Material")
                                    with st.expander("View Study Material"):
                                        st.markdown(study_material)
                                    
                                    # Add a download button for the study material
                                    study_material_text = f"""
                                    Level {level_idx} Study Guide
                                    ========================
                                    
                                    Study Material:
                                    {study_material}
                                    
                                    Review Questions:
                                    """
                                    for wrong in wrong_answers:
                                        study_material_text += f"\n{wrong['question_num']}. {wrong['question']}\n"
                                        study_material_text += f"   Correct Answer: {wrong['correct_answer']}\n"
                                        study_material_text += f"   Explanation: {wrong['explanation']}\n"
                                    
                                    st.download_button(
                                        label="Download Study Material",
                                        data=study_material_text,
                                        file_name=f"level_{level_idx}_study_guide.txt",
                                        mime="text/plain"
                                    )
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.button("Retry Test", key=f"retry_{level_idx}"):
                                        st.session_state[f'user_answers_level_{level_idx}'] = {}
                                        st.rerun()
                                with col2:
                                    if st.button("Generate New Questions", key=f"new_questions_{level_idx}"):
                                        st.session_state[f'mcqs_generated_level_{level_idx}'] = False
                                        st.session_state[f'mcqs_level_{level_idx}'] = None
                                        st.session_state[f'user_answers_level_{level_idx}'] = {}
                                        st.rerun()

        # Show level progress in sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Your Progress")
        for level in range(1, 5):
            if level in st.session_state.levels_completed:
                score_info = st.session_state.level_scores.get(level, {})
                score = score_info.get('score', 0)
                percentage = score_info.get('percentage', 0)
                st.sidebar.success(f"Level {level} ✅ - Score: {score}/10 ({percentage:.1f}%)")
            elif level == st.session_state.current_level:
                st.sidebar.info(f"Level {level} 🔄")
            else:
                st.sidebar.warning(f"Level {level} 🔒")

        # If all levels are completed, show final results
        if len(st.session_state.levels_completed) == 4:
            st.sidebar.markdown("---")
            st.sidebar.header("🏆 Final Results")
            total_score = sum(info['score'] for info in st.session_state.level_scores.values())
            average_percentage = sum(info['percentage'] for info in st.session_state.level_scores.values()) / 4
            st.sidebar.markdown(f"""
            **Total Score:** {total_score}/40
            **Average:** {average_percentage:.1f}%
            """)

# Footer
st.markdown("---")
st.markdown("Video Processing System | Powered by Whisper and LLM")
