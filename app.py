from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer,  AutoTokenizer, AutoModelWithLMHead
from PIL import Image
import requests
from io import BytesIO
import torch
import nltk
from nltk.tokenize import sent_tokenize
import json 

nltk.download('punkt')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# captioning model
captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# summary model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m_name = "marefa-nlp/summarization-arabic-english-news"

summarizer_tokenizer = AutoTokenizer.from_pretrained(m_name)
summarizer_model = AutoModelWithLMHead.from_pretrained(m_name).to(device)
captioning_model.to(device)


def get_summary(text, tokenizer = summarizer_tokenizer, model= summarizer_model, device=device, num_beams=2):
    if len(text.strip()) < 50:
        return ["Please provide a longer text"]
    
    text = "summarize: <paragraph> " + " <paragraph> ".join([ s.strip() for s in sent_tokenize(text) if s.strip() != ""]) + " </s>"
    text = text.strip().replace("\n","")
    
    tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)
    summary_ids = model.generate(
            tokenized_text,
            max_length=512,
            num_beams=num_beams,
            repetition_penalty=1.5, 
            length_penalty=1.0, 
            early_stopping=True
    )
    
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # return summary_ids
    return [ s.strip() for s in output.split("<hl>") if s.strip() != "" ]

max_length = 24
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def get_img_caption(image_urls):
  image_urls = image_urls.split(",")
  images = []
  for img_url in image_urls:
    if 'http' in img_url: # for urls
      response = requests.get(img_url)
      img = Image.open(BytesIO(response.content))
    else: # for path 
      img = Image.open(img_url)

    if img.mode != "RGB":
      img = img.convert(mode="RGB")

    images.append(img)

  pixel_values = captioning_feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = captioning_model.generate(pixel_values, **gen_kwargs)

  preds = captioning_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

def complete_pipeline(query):
    query_list = query.split(":article:")
    img_path, text_article = query_list[0], query_list[1]
    caption = get_img_caption(img_path)
    # text = f'Attached image caption:{caption[0]}. Paragraph:{text_article}'
    summary = get_summary(text_article)
    return f"{caption[0]}summary:{' '.join(summary)}"
    


@app.route('/')
def home():
    return render_template('index.html')

@cross_origin()
@app.route('/summarize', methods=["POST"])
def summarize():
    if request.method == "POST":
        my_req_obj = json.loads(request.data)
        # print(my_req_obj['query'])
        return complete_pipeline(my_req_obj['query'])
    return "ABAB"

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()

