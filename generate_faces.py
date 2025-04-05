import os
from PIL import Image, ImageDraw

# Create folders
os.makedirs("mini_dataset/happy", exist_ok=True)
os.makedirs("mini_dataset/sad", exist_ok=True)

def draw_face(label="happy", filename="face.png"):
    # Create blank white image
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw face outline
    draw.ellipse((8, 8, 56, 56), outline="black", width=2)

    # Eyes
    draw.ellipse((20, 20, 26, 26), fill="black")
    draw.ellipse((38, 20, 44, 26), fill="black")

    # Smile or frown
    if label == "happy":
        draw.arc((20, 30, 44, 50), start=0, end=180, fill="black", width=2)
    else:
        draw.arc((20, 40, 44, 60), start=180, end=360, fill="black", width=2)

    img.save(filename)

# Generate images
for i in range(10):
    draw_face("happy", f"mini_dataset/happy/happy_{i}.png")
    draw_face("sad", f"mini_dataset/sad/sad_{i}.png")

print("✅ Dummy happy/sad faces generated in 'mini_dataset/' folder")
