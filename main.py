from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pytesseract
import re
from starlette.requests import Request
from starlette.responses import Response
from io import BytesIO

app = FastAPI()

myconfig = r"--psm 1"
pattern = r"(\w{2})(\d{16})"
name_pattern = re.compile(r"(([A-Z]{1}[a-z]+)[\s]([A-Z]{1}[a-z]+|([A-Z]?\.))*[\s]?([A-Z]{1}[a-z]+))")
year_pattern = re.compile(r"\b\d{4}\b")

def check_validity(udid_image):
    img = cv2.imdecode(np.fromstring(udid_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    def grey_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grey_image = grey_scale(img)
    threshold, black_white_image = cv2.threshold(grey_image, 150, 130, cv2.THRESH_BINARY)

    def noise_removal(image):   
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)

        return image

    no_noise = noise_removal(black_white_image)
    ocr_result = pytesseract.image_to_string(no_noise, config=myconfig)
    ocr_status = ocr_result.lower()

    if "unique disability id" in ocr_status and "government of india" in ocr_status and re.search(pattern, ocr_status):
        return "Authorized", re.findall(name_pattern, ocr_result)[0][0], re.findall(year_pattern, ocr_result)[0]
    else:
        return "Unauthorized", None, None

@app.post("/verify")
async def verify_udid_image(request: Request):
    form_data = await request.form()
    input_file = form_data["input_file"]
    
    try:
        image_bytes = await input_file.read()
        result, name, year = check_validity(BytesIO(image_bytes))
        
        if result == "Authorized":
            return JSONResponse(content={"status": result, "name": name, "year": year})
        else:
            raise HTTPException(status_code=401, detail="Unauthorized")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
