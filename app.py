import streamlit as st
import os
import sys
from PIL import Image
from pathlib import Path

# Add components to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ocr.handwriting_recognizer import HandwritingRecognizer
from scoring.scoring_agent_wrapper import ScoringWrapper
from feedback.feedback_generator import FeedbackGenerator

# --- Cache Models ---
@st.cache_resource
def load_ocr():
    return HandwritingRecognizer()

@st.cache_resource
def load_scoring():
    return ScoringWrapper()

@st.cache_resource
def load_feedback():
    return FeedbackGenerator()

# --- Page Config ---
st.set_page_config(
    page_title="Bilingual AI Essay Grading System",
    page_icon="📝",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .score-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #007bff;
    }
    .feedback-section {
        background-color: #ffffff;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🌐 Bilingual AI Essay Grading System")
st.markdown("### Professional Grading for English & Arabic Essays")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    lang_mode = st.selectbox("Document Language", ["Auto-Detect", "English", "Arabic"])
    if lang_mode == "Arabic":
        st.info("🔤 Arabic mode: Using EasyOCR for Arabic handwriting recognition.")
    else:
        st.info("🔤 English mode: Using TrOCR for English handwriting recognition. For Arabic, select 'Arabic' in the dropdown.")

# --- UI Components ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input Method")
    tabs = st.tabs(["📝 Paste Text", "📂 Upload File", "🖼️ Handwriting OCR"])
    
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
        
    essay_text = ""
    
    with tabs[0]:
        input_text = st.text_area("Paste your essay here:", height=300, placeholder="Start typing...")
        if input_text.strip():
            essay_text = input_text
            
    with tabs[1]:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file:
            essay_text = uploaded_file.read().decode("utf-8")
            
    with tabs[2]:
        uploaded_img = st.file_uploader("Upload handwritten essay image", type=['jpg', 'jpeg', 'png'])
        if uploaded_img:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("🚀 Extract Text from Image"):
                with st.spinner("Analyzing handwriting (this may take a moment)..."):
                    try:
                        ocr_engine = load_ocr()
                        if lang_mode == "Auto-Detect":
                            st.warning("OCR defaults to English on Auto-Detect. Select 'Arabic' in the sidebar for Arabic.")
                            lang_code = "en"
                        elif lang_mode == "Arabic":
                            lang_code = "ar"
                        else:
                            lang_code = "en"
                            
                        extracted = ocr_engine.extract_text_from_handwriting(image, lang=lang_code)
                        st.session_state.ocr_text = extracted
                        st.success(f"Text extracted successfully! (Language: {'Arabic' if lang_code == 'ar' else 'English'})")
                    except Exception as e:
                        st.error(f"OCR Error: {e}")
        else:
            st.session_state.ocr_text = ""
            
        if st.session_state.ocr_text and not essay_text:
            essay_text = st.session_state.ocr_text

with col2:
    st.subheader("📊 Results & Feedback")
    
    if essay_text:
        st.markdown("**Processed Text Preview:**")
        # For Arabic text, display with RTL direction
        is_arabic = lang_mode == "Arabic" or any('\u0600' <= c <= '\u06ff' for c in essay_text[:100])
        preview_text = essay_text[:500] + "..." if len(essay_text) > 500 else essay_text
        if is_arabic:
            st.markdown(
                f'<div style="direction:rtl; text-align:right; font-size:16px; \'Scheherazade New\', Arial, sans-serif; background-color:#f0f4ff; border-radius:8px; padding:12px;">{preview_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(preview_text)
        
        if st.button("📈 Grade Essay"):
            with st.spinner("Grading essay and generating feedback..."):
                # Load Engines
                scorer = load_scoring()
                feedback_engine = load_feedback()
                
                # Detect Language
                if lang_mode == "Auto-Detect":
                    lang = feedback_engine.detect_language(essay_text)
                else:
                    lang = lang_mode.lower()
                
                # Score
                score = scorer.score_essay(essay_text, lang=lang)
                
                # Feedback
                results = feedback_engine.generate_feedback(essay_text, score)
                
                # Display Score
                st.markdown(f"""
                    <div class="score-box">
                        <p style='font-size: 24px; margin-bottom: 0;'>Final Score</p>
                        <p style='font-size: 64px; font-weight: bold; color: #007bff; margin-top: 0;'>{results['score']}/10</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display Feedback Sections
                st.markdown("#### Detailed Analysis")
                
                fb = results['feedback']
                sections = {
                    "Grammar": fb['grammar'],
                    "Content": fb['content'],
                    "Structure": fb['structure'],
                    "Vocabulary": fb['vocabulary'],
                    "Suggestions": fb['suggestions']
                }
                
                for title, content in sections.items():
                    st.markdown(f"""
                        <div class="feedback-section">
                            <strong>{title}</strong><br>
                            {content}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Download Report
                report_text = f"Bilingual AI Essay Report\n"
                report_text += f"{'='*30}\n"
                report_text += f"Score: {results['score']}/10\n"
                report_text += f"Language: {lang.capitalize()}\n\n"
                report_text += "Feedback:\n"
                for k, v in fb.items():
                    report_text += f"- {k.capitalize()}: {v}\n"
                
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_text,
                    file_name="essay_feedback_report.txt",
                    mime="text/plain"
                )
    else:
        st.write("Enter an essay on the left to see results.")

# --- Footer ---
st.divider()
st.caption("AES System v2.0 | Powered by Cross-Attention Transformer | Bilingual Support (EN/AR)")
