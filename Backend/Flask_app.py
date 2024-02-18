from flask import Flask, render_template, request,jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import *
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import os
import base64
from flask_cors import CORS, cross_origin
import openai
from openai import OpenAI



app = Flask(__name__)
CORS(app)
@app.route("/upload",methods = ['GET', 'POST'])
def home():
    def video_summarizer(video,audio_1):
        video = VideoFileClip(video)
        audio = video.audio 
        audio.write_audiofile(audio_1)
        result = pipe(audio_1)
        return result["text"]
    

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    torch.cuda.empty_cache()
    model_id = "openai/whisper-base"

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

    data = request.get_json()
    if("video" in data):
        if data["video"].startswith('data:video'):
            data["video"] = data["video"].split(',')[1]

        # Decode the base64 string
        video_data = base64.b64decode(data["video"])

        # Write the decoded bytes to a file
        with open("Videos/video.mp4", 'wb') as f:
            f.write(video_data)

        text = video_summarizer("Videos/video.mp4","Videos/audio.mp3")
        file_path = 'API_KEY.txt'
        with open(file_path, 'r') as file:
            api_key = file.read().strip() 
        openai.api_key = api_key
        client = OpenAI(api_key=api_key)      

        response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text+" Summarize this text based on the following: Highlight the summary as bullet points and the break the summary using various headings"}])        

        return jsonify(response.choices[0].message.content)
    
    if("PDF" in data):
        
    # app.config['UPLOAD_FOLDER']="./Videos"
    # f = request.files['file']
    # f.filename = "Video.mp4"
    # f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    #text = video_summarizer(video,audio_1)

if __name__ == "__main__":
    app.run(debug=True)
