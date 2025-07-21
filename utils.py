import whisper
import torchaudio
import tempfile
import os
import subprocess
import moviepy.editor as mp
from transformers import pipeline
from difflib import SequenceMatcher

# Load Whisper model once
model = whisper.load_model("base")


# --------- Audio Extraction & Transcription ---------

def extract_audio(video_path):
    audio_path = "temp_audio.wav"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.numel() == 0:
            raise ValueError("Audio seems empty.")
    except Exception as e:
        raise RuntimeError("❌ Failed to load audio. It might be corrupted or silent.") from e

    result = model.transcribe(audio_path, verbose=False)
    text = result["text"]
    segments = result.get("segments", [])
    return text, segments

def extract_audio_and_transcribe(video_path):
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    result = model.transcribe(audio_path)
    return result['text']


# --------- Summarization Utilities ---------

def summarize_text(text):
    if len(text) > 8000:
        text = text[:8000]
    summarizer = pipeline("summarization", model="Falconsai/text_summarization", tokenizer="Falconsai/text_summarization")
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n"
    return summary

def generate_summary(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# --------- Video Clipping Utilities ---------

def clip_video(video_path, start_time, end_time, output_path="short_clip.mp4"):
    clip = mp.VideoFileClip(video_path).subclip(start_time, end_time)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    return output_path

def match_summary_to_segments(summary, segments):
    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    matched_segments = []
    for seg in segments:
        seg_text = seg.get("text", "")
        if similar(summary, seg_text) > 0.3 or any(word in seg_text.lower() for word in summary.lower().split()):
            matched_segments.append((seg['start'], seg['end']))

    if not matched_segments:
        return []

    merged_segments = []
    current_start, current_end = matched_segments[0]

    for start, end in matched_segments[1:]:
        if start - current_end <= 2:
            current_end = end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end

    merged_segments.append((current_start, current_end))
    return merged_segments

def get_summary_video(video_path, transcript, summary_text):
    video = mp.VideoFileClip(video_path)
    segments = []
    sentences = transcript.split(". ")
    summary_sentences = summary_text.split(". ")

    duration_per_sentence = video.duration / len(sentences) if sentences else video.duration

    for i, sentence in enumerate(sentences):
        for summ in summary_sentences:
            ratio = SequenceMatcher(None, sentence.lower(), summ.lower()).ratio()
            if ratio > 0.5:
                start = i * duration_per_sentence
                end = min((i + 1) * duration_per_sentence, video.duration)
                segments.append(video.subclip(start, end))
                break

    if segments:
        final_clip = mp.concatenate_videoclips(segments)
        output_path = os.path.join(tempfile.gettempdir(), "summary_video.mp4")
        final_clip.write_videofile(output_path, codec="libx264")
        return output_path
    else:
        return None