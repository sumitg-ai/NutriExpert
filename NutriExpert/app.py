from flask import Flask, request, render_template, send_file
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import requests
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from datetime import datetime

# Define paths
#MODEL_PATH = "/mnt/c/Marriott/Learning/AI Projects/NutritionExpert_V2/Finetuned_Models/v3/indian_food_model_epoch_3.pth"
#LABELS_PATH = "/mnt/c/Marriott/Learning/AI Projects/NutritionExpert_V2/Finetuning/indian_food_labels.json"

MODEL_PATH = os.path.join("model", "indian_food_model_epoch_3.pth")
LABELS_PATH = os.path.join("model", "indian_food_labels.json")

app = Flask(__name__)

# Load class labels
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

# Load model
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_path):
    image_tensor = transform_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
    return LABELS[predicted.item()]

def get_nutritional_info(food_item):
    app_id = '4657f251'
    app_key = '66f28ceff5252fdd5723768e9c96aa44'
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

    headers = {
        'x-app-id': app_id,
        'x-app-key': app_key,
        'Content-Type': 'application/json'
    }

    data = {"query": food_item}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        data = response.json()

        if data.get('foods'):
            nutritional_info = data['foods'][0]
            return {
                'Food Item': nutritional_info.get('food_name', 'N/A'),
                'Calories': nutritional_info.get('nf_calories', 'N/A'),
                'Protein': nutritional_info.get('nf_protein', 'N/A'),
                'Fiber': nutritional_info.get('nf_dietary_fiber', 'N/A'),
                'Fat': nutritional_info.get('nf_total_fat', 'N/A'),
                'Carbs': nutritional_info.get('nf_total_carbohydrate', 'N/A')
            }
        else:
            print("No foods found in the response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching nutritional info: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = "static/uploads/" + file.filename
            file.save(filepath)
            prediction = predict(filepath)
            nutrition = get_nutritional_info(prediction)

            return render_template("result.html", 
                                   food_name=prediction,
                                   image_url=f"/static/uploads/{file.filename}",
                                   calories=nutrition.get('Calories', 'N/A'),
                                   protein=nutrition.get('Protein', 'N/A'),
                                   fat=nutrition.get('Fat', 'N/A'),
                                   carbs=nutrition.get('Carbs', 'N/A'),
                                   fiber=nutrition.get('Fiber', 'N/A'))
    return render_template("upload.html")

@app.route("/diet-plan", methods=["GET"])
def diet_plan_form():
    return render_template("diet_plan.html")

@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    sections = request.form.to_dict(flat=False)
    client_name = request.form.get("client_name", "Client")
    date_str = datetime.now().strftime("%B %d, %Y")

    parsed = []
    for key in sections:
        if "section[" in key:
            section_index = key.split("[")[1].split("]")[0]
            if len(parsed) <= int(section_index):
                parsed.append({"time": "", "options": []})
            if "time" in key:
                parsed[int(section_index)]["time"] = sections[key][0]
            elif "option" in key:
                parsed[int(section_index)]["options"] = sections[key]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("HealthifyMe - Diet Plan", styles["Title"]))
    elements.append(Paragraph(f"For: {client_name}", styles["Heading2"]))
    elements.append(Paragraph(f"Date: {date_str}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    for section in parsed:
        elements.append(Paragraph(f"<b>{section['time']}</b>", styles["Heading3"]))
        elements.append(Spacer(1, 6))
        for idx, option in enumerate(section["options"]):
            elements.append(Paragraph(f"Option {idx+1}  {option}", styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, download_name="diet_plan.pdf", mimetype='application/pdf')

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
