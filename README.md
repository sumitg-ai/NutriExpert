# 🍽️ NutriExpert – AI-Driven Nutrition Assistant

**NutriExpert** is an AI-powered nutrition assistant that identifies Indian food items from uploaded images, retrieves their caloric and macronutrient information using the Nutritionix API, and generates personalized diet-plan PDFs.

It combines computer vision (fine-tuned ResNet50 model) with nutritional intelligence to help users make informed dietary choices.

---

## 🚀 Features

- 🍛 **Food Recognition** – Detects Indian dishes from uploaded images using a fine-tuned CNN (ResNet50).
- 🔢 **Nutrition Analysis** – Fetches calories, proteins, fats, and carbs via the Nutritionix API.
- 📄 **Diet Plan Export** – Generates customizable diet plan PDFs with nutritional breakdowns.
- ⚡ **FastAPI Backend** – Lightweight, production-ready API backend.
- 🧠 **Pretrained Model Included** – `indian_food_model_epoch_3.pth` and `indian_food_labels.json`.

---

## 🧩 Tech Stack

- **Python 3.8+**
- **FastAPI** (backend framework)
- **PyTorch** (for CNN model inference)
- **Nutritionix API** (nutrition data)
- **ReportLab** (PDF generation)
- **Uvicorn / Gunicorn** (server deployment)

---

## 📦 Project Structure

```
NutriExpert/
│
├── app.py                     # Main FastAPI app
├── requirements.txt           # Python dependencies
├── indian_food_model_epoch_3.pth   # Trained ResNet50 model
├── indian_food_labels.json    # Food class labels
├── templates/                 # HTML templates (UI + PDF)
├── static/                    # Static assets
├── Procfile                   # For Heroku deployment
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sumitg-ai/NutriExpert.git
cd NutriExpert
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
# Activate
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up environment variables

Create a `.env` file in the project root and add:
```
NUTRITIONIX_API_KEY=your_api_key
NUTRITIONIX_APP_ID=your_app_id
SECRET_KEY=your_secret_key
```

> You can get free Nutritionix credentials here: https://developer.nutritionix.com/

### 5️⃣ Run the app
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at 👉 **http://localhost:8000**

---

## 🧠 How It Works

1. Upload a food image through the web UI or API.  
2. The system:
   - Preprocesses the image (resize, normalize)
   - Predicts the dish using the fine-tuned **ResNet50** model  
   - Maps the predicted class ID → label from `indian_food_labels.json`
   - Queries **Nutritionix API** for calories and macros
   - Displays results and optionally generates a **PDF diet plan**
3. Download the PDF or integrate via API endpoints.

---

## 🧪 Example API Call

```bash
curl -X POST -F "file=@/path/to/dish.jpg" http://localhost:8000/predict
```

**Sample Response:**
```json
{
  "predicted_label": "Paneer Butter Masala",
  "confidence": 0.94,
  "calories": 310,
  "protein": 12,
  "fat": 22,
  "carbohydrates": 10
}
```

---

## 🧾 PDF Export

- Generated diet-plan PDFs include:
  - Dish name & image
  - Calorie & macro breakdown
  - Daily diet plan summary
- Uses **ReportLab** for high-quality export.

---

## ☁️ Deployment

### 🔹 Deploy to Heroku
```bash
heroku create nutri-expert
heroku config:set NUTRITIONIX_API_KEY=your_api_key NUTRITIONIX_APP_ID=your_app_id
git push heroku main
```

### 🔹 Docker Deployment
(Optional)
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:
```bash
docker build -t nutri-expert .
docker run -p 8000:8000 nutri-expert
```

---

## 🔮 Roadmap

- [ ] Support multiple dishes in one image (object detection)
- [ ] Add user profiles with saved diet plans
- [ ] Integrate AI diet recommendations (based on health goals)
- [ ] Cloud model hosting & inference acceleration

---

## 🧑‍💻 Author

**Sumit Ghosh**  
AI Developer | MLOps | Applied AI Systems  
[GitHub](https://github.com/sumitg-ai) • [LinkedIn](https://linkedin.com/in/sumit-ghosh-ai)

---

## 🪪 License

MIT License © 2025 [Sumit Ghosh](https://github.com/sumitg-ai)

---

## 🌟 Acknowledgements

- [Indian Food Image Dataset (Kaggle)](https://www.kaggle.com/datasets/ps2004/food-dataset)
- [Nutritionix API](https://developer.nutritionix.com/)
- [PyTorch](https://pytorch.org/)
