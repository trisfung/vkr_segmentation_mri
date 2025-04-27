from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid
import pathlib

# Определяем абсолютные пути, базируясь на текущем расположении main.py
BASE_DIR = pathlib.Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load YOLO model once
model = YOLO("c:/users/user/desktop/vkr_seg/runs/segment/train/weights/best.pt")

# Создаем директории, если они не существуют
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_img": None, "error": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, file: UploadFile = File(...)):
    error = None
    result_img = None
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            error = "Файл должен быть изображением!"
            return templates.TemplateResponse("index.html", {"request": request, "result_img": None, "error": error})
        # Save uploaded file
        file_ext = os.path.splitext(file.filename)[-1]
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        upload_path = os.path.join(str(UPLOAD_DIR), unique_name)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Run YOLO prediction with minimum confidence 0.5
        results = model.predict(source=upload_path, conf=0.5, save=True, project=str(RESULTS_DIR), name=unique_name, exist_ok=True)
        # Find result image (YOLO saves to a subfolder)
        result_folder = os.path.join(str(RESULTS_DIR), unique_name)
        # Find first image in result folder
        result_files = [f for f in os.listdir(result_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not result_files:
            error = "Не удалось получить результат обработки YOLO."
            return templates.TemplateResponse("index.html", {"request": request, "result_img": None, "error": error})
        result_img = f"/static/results/{unique_name}/{result_files[0]}"
    except Exception as e:
        error = f"Ошибка обработки: {str(e)}"
    return templates.TemplateResponse("index.html", {"request": request, "result_img": result_img, "error": error}) 