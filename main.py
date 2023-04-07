###=== Server ===###
import os
import requests
from PIL import Image
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

app = FastAPI()

class UploadImageInput(BaseModel):
    imageLink: str
    userId: str

class ImgCaptionInput(BaseModel):
    userId: str

class VisualQuestionAnswerInput(BaseModel):
    userId: str
    question: str


class ImageCaptioning:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)
        self.model.config.max_new_tokens = 128  # Set max_new_tokens

    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions

class VisualQuestionAnswering:
    def __init__(self, device):
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)
        self.model.config.max_new_tokens = 128  # Set max_new_tokens

    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer
"""
1. The image is opened using the Python Imaging Library (PIL).
2. The image is resized to fit within a 512x512 bounding box while maintaining its aspect ratio. The new width and height are rounded to the nearest multiple of 64.
3. The image is converted to the RGB color space if it's not already in that format.
4. The resized and converted image is saved as a PNG file with a unique filename in the 'image' directory.
"""
def save_and_process_image(image_path, user_id):
    image_filename = os.path.join('image', f"{user_id}.png")
    os.makedirs('image', exist_ok=True)
    img = Image.open(image_path)
    width, height = img.size
    ratio = min(512 / width, 512 / height)
    width_new, height_new = (round(width * ratio), round(height * ratio))
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64
    img = img.resize((width_new, height_new))
    img = img.convert('RGB')
    img.save(image_filename, "PNG")
    return image_filename

def download_image(image_url, user_id):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_path = os.path.join('image', f"{user_id}.png")
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    else:
        raise HTTPException(status_code=400, detail="Image download failed")

# Create the 'image' folder if it doesn't exist
os.makedirs('image', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_captioning = ImageCaptioning(device=device)
visual_question_answering = VisualQuestionAnswering(device=device)

#== POST APIs ==#
@app.post("/uploadImage")
async def upload_image(input_data: UploadImageInput):
    image_url = input_data.imageLink
    user_id = input_data.userId
    image_path = download_image(image_url, user_id)
    processed_image_path = save_and_process_image(image_path, user_id)
    return {"status": "success", "message": "Image uploaded and processed"}

@app.post("/imgCaptionApi")
async def img_caption_api(input_data: ImgCaptionInput):
    user_id = input_data.userId
    image_path = os.path.join('image', f"{user_id}.png")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    captions = image_captioning.inference(image_path)
    return {"captions": captions}

@app.post("/visualQuestionAnswerApi")
async def visual_question_answer_api(input_data: VisualQuestionAnswerInput):
    user_id = input_data.userId
    question = input_data.question
    image_path = os.path.join('image', f"{user_id}.png")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    inputs = f"{image_path},{question}"
    answer = visual_question_answering.inference(inputs)
    return {"answer": answer}

#== GET API ==#
@app.get("/uploadImage")
async def upload_image(imageLink: str, userId: str):
    image_url = imageLink
    user_id = userId
    image_path = download_image(image_url, user_id)
    processed_image_path = save_and_process_image(image_path, user_id)
    return {"status": "success", "message": "Image uploaded and processed"}

@app.get("/imgCaptionApi")
async def img_caption_api(userId: str):
    user_id = userId
    image_path = os.path.join('image', f"{user_id}.png")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    captions = image_captioning.inference(image_path)
    return {"captions": captions}

@app.get("/visualQuestionAnswerApi")
async def visual_question_answer_api(userId: str, question: str):
    user_id = userId
    image_path = os.path.join('image', f"{user_id}.png")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    inputs = f"{image_path},{question}"
    answer = visual_question_answering.inference(inputs)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)