import streamlit as st
import numpy as np
import torch
import os
import tempfile
import moviepy.editor as mp

from utils import (
    extract_audio,
    transcribe_audio,
    summarize_text,
    clip_video,
    match_summary_to_segments,
    extract_audio_and_transcribe,
    generate_summary,
    get_summary_video
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎥",
    layout="wide",
)

# ---- Header ----
st.markdown("""
    <div style="background-color:#1f2937;padding:20px;border-radius:10px">
        <h1 style="color:#faclearcc15;text-align:center;">🎥 AI Video Summarizer</h1>
        <h4 style="color:#f3f4f6;text-align:center;">Smart Summaries & Video Clipping for Long Lectures</h4>
    </div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.markdown("### 💡 Quick Guide")
st.sidebar.success("""
📥 **Step 1:** Upload your video in **Tab 1**

📝 **Step 2:** Generate transcript & summary in **Tab 2**

🎬  **Step 3:** Extract manual clip or AI summary clip in **Tab 3**
""")

st.markdown("---")
st.sidebar.markdown("### 📝 About")
st.sidebar.success("Made by **Kiran, Komal, Preeti**")

# ---- Main Tabs ----
tab1, tab2, tab3 = st.tabs(["📥 Upload & Process", "📝 Transcript & Summary", "🎞 Video Clips"])

with tab1:
    st.subheader("Upload and Process Video")

    video_file = st.file_uploader("Upload a lecture video (MP4/MOV/AVI/WEBM)", type=["mp4", "mov", "avi", "webm"])

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name
            st.session_state['video_path'] = video_path

        st.video(video_path)
        st.success("✅ Video Uploaded Successfully!")

        if st.button("🔊 Extract & Transcribe Audio"):
            st.info("Processing audio and transcription...")
            audio_path = extract_audio(video_path)
            transcript, segments = transcribe_audio(audio_path)
            st.session_state['transcript'] = transcript
            st.session_state['segments'] = segments
            st.success("✅ Transcription Complete!")


with tab2:
    st.subheader("Transcript & Summary")

    if 'transcript' in st.session_state:
        st.text_area("📝 Transcript", st.session_state['transcript'], height=300)

        if st.button("🧠 Generate Summary"):
            summary = summarize_text(st.session_state['transcript'])
            st.session_state['summary'] = summary
            st.success("✅ Summary Generated!")

        if 'summary' in st.session_state:
            st.text_area("🧾 Summary", st.session_state['summary'], height=200)

            st.download_button("📥 Download Transcript", st.session_state['transcript'], file_name="transcript.txt")
            st.download_button("📥 Download Summary", st.session_state['summary'], file_name="summary.txt")
    else:
        st.info("Upload and transcribe a video first from the first tab.")


with tab3:
    st.subheader("🎬 Video Clip Tools")

    if 'video_path' in st.session_state:

        st.markdown("#### ✂️ Extract a Clip from Uploaded Video")
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start time (seconds)", min_value=0)
        with col2:
            end_time = st.number_input("End time (seconds)", min_value=1)

        if st.button("✂️ Create Clip"):
            if start_time < end_time:
                clip_path = clip_video(st.session_state['video_path'], start_time, end_time)
                st.video(clip_path)
                st.success("✅ Clip Created Successfully!")
            else:
                st.error("⚠️ Start time must be less than end time.")

        st.markdown("---")
        st.markdown("#### 🎞 AI Summary Clip Generator")

        if st.button("🎞 Generate AI Summary Clip"):
            with st.spinner("Processing AI Summary Clip... This may take a few minutes."):
                transcript2 = extract_audio_and_transcribe(st.session_state['video_path'])
                summary_video2 = get_summary_video(st.session_state['video_path'], transcript2, st.session_state['summary']) if 'summary' in st.session_state else None

            if summary_video2:
                st.subheader("Summary Clip Video")
                st.video(summary_video2)
                st.success("✅ AI Summary Clip Generated Successfully!")
            else:
                st.warning("⚠️ No matching segments found for the summary.")
    else:
        st.info("Upload a video first from the first tab.")

st.markdown("---")