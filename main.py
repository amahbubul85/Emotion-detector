from PIL import Image
from utils import load_model, predict_emotion

img = Image.open("ah_face.jpg").convert('RGB')
model = load_model()
label, prob = predict_emotion(model, img)
print(f"Prediction: {label} (confidence: {prob:.2f})")
