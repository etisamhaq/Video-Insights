# 🎥 Video Transcription & Analysis App  

This **Application** enables users to **upload videos or provide local paths** for transcription and analysis.  
It utilizes **OpenAI’s Whisper** for **audio-to-text conversion** and integrates **LangChain** for:  

✅ **Summarization**  
✅ **Key Points Extraction**  
✅ **Interactive Chat**  
✅ **Question Generation & Evaluation**  

![pic](https://github.com/etisamhaq/Video-Insights/blob/main/pic1.jpg)

The app evaluates user responses, determines correctness, and provides **correct answers with explanations**.  
It features **progress indicators**, **error handling**, and a clean **tab-based UI** for a seamless experience.  

---

## ✨ Features  

🔹 **Upload or Provide Local Video Paths** – Process videos effortlessly.  
🔹 **Whisper-Based Transcription** – Convert speech to text with high accuracy.  
🔹 **Summarization & Key Points Extraction** – Get concise insights from video content.  
🔹 **Interactive Chat** – Engage in AI-powered discussions about the video.  
🔹 **Question Generation & Evaluation** – Automatically generate questions and assess user responses.  
🔹 **Correct Answer & Explanation** – Provide the right answer along with an explanation if the response is incorrect.  
🔹 **Progress Indicators & Error Handling** – Ensures a smooth user experience.  

---

## 🏗️ Tech Stack  

| Technology  | Purpose |
|------------|---------|
| **Python**  | Core programming language |
| **Streamlit** | Web interface |
| **OpenAI Whisper** | Audio transcription |
| **LangChain** | LLM-based summarization, Q&A, and chat |
| **Groq API & Llama model** | Used for chat and analysis |

---

## 🚀 Installation & Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/etisamhaq/Video-Insights
cd Video-Insights

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 3️⃣ Run the Application
```bash
streamlit run app.py

### 4️⃣ Upload or Provide a Video Path
🔹 Upload a video file or enter a local file path.
🔹 The app will transcribe, analyze, and generate questions.
