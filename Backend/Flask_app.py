from flask import Flask, request,jsonify
# from flask import send_from_directory
# from werkzeug.utils import secure_filename
from moviepy.editor import *
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import base64
from flask_cors import CORS, cross_origin
import openai
from openai import OpenAI
import fitz


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
    
    def extract_text_from_pdf_ignoring_footers(pdf_path, footer_threshold=50):
        """
        Extract text from a PDF, excluding footers based on a vertical threshold from the bottom of the page.
        Print the slide (page) number before the content of that slide.

        :param pdf_path: Path to the PDF file.
        :param footer_threshold: Vertical distance from the bottom to identify and exclude footers.
        """
        document = fitz.open(pdf_path)
        full_text = ""  # Initialize a variable to hold all text, for printing or other use

        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_rect = page.rect
            # Adjust content_region to exclude the footer area
            #content_region = fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1 - footer_threshold)

            # Extract text from the defined content region, excluding the footer
            page_text = page.get_text()

            # Format slide/page number text
            slide_text = f"Slide {page_num + 1}:\n{page_text}\n"
            print(slide_text)  # Print the slide number and its content

            full_text += slide_text  # Optionally concatenate text for further use

        document.close()
        return full_text


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
    file_path = 'API_KEY.txt'
    with open(file_path, 'r') as file:
        api_key = file.read().strip() 

    openai.api_key = api_key
    client = OpenAI(api_key=api_key)  

    def api_calls(prompt,text):
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt+" "+text+" Summarize this text based on the following: Highlight the summary as bullet points and the break the summary using various headings"}]) 
        
        return response.choices[0].message.content

    def word_splitter(text):
        video_text_array = [word for word in text.split()]
        word_limit = 2000
        num_chunks = len(video_text_array) // word_limit + (1 if len(video_text_array) % word_limit else 0)
    
        # Create chunks using list slicing
        chunks = [' '.join(video_text_array[i * word_limit : (i + 1) * word_limit]) for i in range(num_chunks)]

        return chunks        

    data = request.get_json()
    if(("video" in data and data["video"]) and "PDF" in data and data["PDF"]):
        if data["video"].startswith('data:video'):
            data["video"] = data["video"].split(',')[1]

        # Decode the base64 string
        video_data = base64.b64decode(data["video"])

        # Write the decoded bytes to a file
        with open("Videos/video.mp4", 'wb') as f:
            f.write(video_data)

        video_text = video_summarizer("Videos/video.mp4","Videos/audio.mp3")

        chunks = word_splitter(video_text)

        response = ""
        for chunk in chunks:
            response+=api_calls("Transcript:",chunk)

        base64_data = data["PDF"].split(',', 1)[-1]

        # Remove any potential unwanted characters (e.g., newlines, spaces)
        base64_data_cleaned = base64_data.replace("\n", "").replace(" ", "")

        # Decode the cleaned base64 string
        decoded_data = base64.b64decode(base64_data_cleaned)

        with open("PDFs/text.pdf", 'wb') as pdf_file:
            pdf_file.write(decoded_data)
        
        pdf_text = extract_text_from_pdf_ignoring_footers("PDFs/text.pdf")

        response+="\n"
        chunks = word_splitter(pdf_text)
        for chunk in chunks:
            response+=api_calls("PDF:",chunk)
  
        res = {
            "msg": response
        }    

        return res        


    if("video" in data and data["video"]):
        if data["video"].startswith('data:video'):
            data["video"] = data["video"].split(',')[1]

        # Decode the base64 string
        video_data = base64.b64decode(data["video"])

        # Write the decoded bytes to a file
        with open("Videos/video.mp4", 'wb') as f:
            f.write(video_data)

        text = video_summarizer("Videos/video.mp4","Videos/audio.mp3")

        chunks = word_splitter(text)

        response = ""
        for chunk in chunks:
            response+=api_calls("Transcript:",chunk)    
        res = {
            "msg": response
        }    

        return res
    
    if("PDF" in data and data["PDF"]):
        base64_data = data["PDF"].split(',', 1)[-1]

        # Remove any potential unwanted characters (e.g., newlines, spaces)
        base64_data_cleaned = base64_data.replace("\n", "").replace(" ", "")

        # Decode the cleaned base64 string
        decoded_data = base64.b64decode(base64_data_cleaned)

        with open("PDFs/text.pdf", 'wb') as pdf_file:
            pdf_file.write(decoded_data)
        
        full_text = extract_text_from_pdf_ignoring_footers("PDFs/text.pdf")
  

        chunks = word_splitter(full_text)

        response = ""
        for chunk in chunks:
            response+=api_calls("PDF:",chunk)    
        res = {
            "msg": response
        }  

        return res
        

if __name__ == "__main__":
    app.run(debug=True)
