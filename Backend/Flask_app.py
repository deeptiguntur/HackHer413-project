from flask import Flask, render_template, request,jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import *
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import os

def video_summarizer(video,audio_1):
    video = VideoFileClip(video)
    audio = video.audio 
    audio.write_audiofile(audio_1)
    result = pipe(audio_1)
    return result["text"]

app = Flask(__name__)

@app.route("/upload",methods = ['GET', 'POST'])
def home():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    torch.cuda.empty_cache()
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]

    app.config['UPLOAD_FOLDER']="./Videos"
    f = request.files['file']
    f.filename = "Video.mp4"
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    text = video_summarizer(video,audio_1)
    return 

if __name__ == "__main__":
    app.run(debug=True)
