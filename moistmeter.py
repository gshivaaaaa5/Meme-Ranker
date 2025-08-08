from PIL import Image
import pytesseract
from textblob import TextBlob
from transformers import CLIPProcessor, CLIPModel
import torch

# ⛏️ Set this if tesseract is not auto-detected (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 📷 Load meme image
meme_path = "C:/Users/hp/moistmeter/images.jpeg"

try:
    image = Image.open(meme_path)
except FileNotFoundError:
    print("❌ Error: meme.jpg not found!")
    exit()

# 🔍 OCR - Extract text from meme
text = pytesseract.image_to_string(image)
print("\n📝 Extracted Text:", text.strip())

# 😐 Sentiment Analysis
blob = TextBlob(text)
sentiment = blob.sentiment.polarity  # Range: -1 (neg) to +1 (pos)
print("💬 Sentiment Score:", sentiment)

# 🎭 Meme template detection using CLIP
template_names = [
    "Drake meme", "Distracted Boyfriend", "Two Buttons",
    "Mocking Spongebob", "Change My Mind", "Gru Plan", "Expanding Brain"
]

print("\n🖼 Detecting Meme Template... (This may take a moment)")

# 🔌 Load CLIP model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 👀 Compare image with text prompts
inputs = processor(text=template_names, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)[0]  # Probabilities for each template

# 🥇 Pick best match
best_template_idx = torch.argmax(probs).item()
best_template = template_names[best_template_idx]
template_confidence = probs[best_template_idx].item()

print(f"🏷️ Detected Template: {best_template} ({template_confidence * 100:.2f}%)")

# 📊 Final Meme Rank Logic (out of 10)
# Weighted combo: 50% sentiment + 50% template match
final_score = ((sentiment + 1) / 2 + template_confidence) / 2 * 10  # Normalize sentiment to [0,1]
final_score = round(final_score, 1)

print(f"\n⭐ Final Meme Rank: {final_score}/10")
