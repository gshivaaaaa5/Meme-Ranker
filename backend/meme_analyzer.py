from PIL import Image
import pytesseract
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from textblob import TextBlob
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust if needed

# Load models globally (faster)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

template_names = [
    "Drake meme", "Distracted Boyfriend", "Two Buttons", "Mocking Spongebob",
    "Change My Mind", "Gru Plan", "Expanding Brain", "Surprised Pikachu",
    "Is This a Pigeon", "Woman Yelling at Cat", "Leonardo DiCaprio Laughing",
    "This Is Fine", "You Guys Are Getting Paid?", "Trade Offer", "Galaxy Brain"
]

def get_moist_comment(score):
    if score > 85: return "Absolute banger. Certified moist."
    if score > 70: return "Solid meme, hits just right."
    if score > 50: return "It's alright. Not bad, not great."
    if score > 30: return "Kind of dry, honestly."
    return "Garbage fire. Bone dry."

def analyze_meme(file):
    image = Image.open(io.BytesIO(file.read()))

    # OCR
    text = pytesseract.image_to_string(image)
    text_short = text[:512]

    # Sentiment
    try:
        result = sentiment_pipeline(text_short)[0]
        label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
        label = label_map.get(result["label"], result["label"])
        sentiment_score = {"POSITIVE": 1.0, "NEUTRAL": 0.5, "NEGATIVE": 0.0}.get(label, 0.5)
    except:
        blob = TextBlob(text)
        sentiment_score = (blob.sentiment.polarity + 1) / 2
        label = "NEUTRAL"

    # CLIP detection
    inputs = clip_processor(text=template_names, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    best_idx = torch.argmax(probs).item()
    best_template = template_names[best_idx]
    confidence = probs[best_idx].item()

    # Moist Score
    moist_score = round((confidence * 0.6 + sentiment_score * 0.4) * 100, 1)
    moist_comment = get_moist_comment(moist_score)

    return {
        "moist_score": moist_score,
        "moist_comment": moist_comment,
        "extractedText": text.strip(),
        "sentimentLabel": label,
        "templateName": best_template,
        "templateConfidence": confidence
    }
