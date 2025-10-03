import time
import cv2
import numpy as np
import pytesseract
import re
import json
import os
from flask import Flask, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)  # folder of app.py
IMAGE_NAME = "a2.jpg"  # hardcoded image
IMAGE_PATH = os.path.join(BASE_DIR, IMAGE_NAME)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs-day2")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

REGEX_PATTERNS={
    "PAN":re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$"),
    "AADHAR":re.compile(r"^\d{4}\s\d{4}\s\d{4}$"),
    "AADHAR_NONSPACE":re.compile(r"^\d{12}$"),
    "DL":re.compile(r"^[A-Z]{3}[0-9]{7}$"),
    "VOTER":re.compile(r"^[A-Z]{3}[0-9]{7}$"),
    "PASSPORT":re.compile(r"^[A-Z][0-9]{7}$"),
    "IFSC":re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
}
DOC_RULES={
    "PAN CARD":{
        "keywords":["PERMANENT ACCOUNT NUMBER", "INCOME TAX DEPARTMENT", "PAN"],
        "front":["INCOME TAX", "PAN", "GOVERNMENT OF INDIA", "NAME", "FATHER", "DATE OF BIRTH"],
        "back":["QR CODE", "VERIFY AUTHENTICITY", "NSDL", "UTIITSL"],
        "patterns":["PAN"]
    },
    "AADHAR CARD":
    {
        "keywords":["AADHAAR", "GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION"],
        "front":["GOVERNMENT OF INDIA", "AADHAAR", "DOB", "GENDER", "NAME"],
        "back":["ADDRESS", "DISTRICT", "STATE", "PIN"],
        "patterns":["AADHAR","AADHAR_NONSPACE"]

    },
    "VOTER ID CARD":{
        "keywords":["ELECTION COMMISSION OF INDIA", "VOTER ID"],
        "front":["ELECTION COMMISSION OF INDIA", "NAME", "DOB", "FATHER"],
        "back":["ADDRESS", "DISTRICT", "STATE", "PIN"],
        "patterns":["VOTER"]

    },
    "PASSPORT":{
        "keywords":["PASSPORT", "REPUBLIC OF INDIA"],
        "front":["PASSPORT", "NAME", "NATIONALITY", "DATE OF BIRTH"],
        "back":["ADDRESS", "EMERGENCY CONTACT"],
        "patterns":["PASSPORT"]

    },
    "BANK PASSBOOK":{
        "keywords":["ACCOUNT", "IFSC", "SAVINGS", "CURRENT"],
        "front":["ACCOUNT NUMBER", "IFSC", "BRANCH", "CUSTOMER ID"],
        "back":["DEPOSIT", "WITHDRAWAL", "BALANCE"],
        "keywords":["IFSC"]

    },
}
def extract_text_blocks(image,draw_boxes=True):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    dilated=cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)
    contours,_=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    extracted=[]
    for contour in sorted(contours,key=lambda ctr:cv2.boundingRect(ctr)[1]):
        x,y,w,h=cv2.boundingRect(contour)
        roi=image[y:y+h,x:x+w]
        text=pytesseract.image_to_string(roi,config="--psm 6").strip()
        if text:
            extracted.append(text)
            if draw_boxes:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return extracted,image #output and bounding box

def detect_document_type(blocks):
    text_all=" ".join(blocks).upper()
    for doc,rules in DOC_RULES.items():
        if any(kw in text_all for kw in rules ["keywords"]):
            return doc
        for pattern_ans in rules["patterns"]:
            if any(REGEX_PATTERNS[pattern_ans].match(b.replace(" " , "")) for b in blocks):
                return doc
    return "-- UNKNOWN DOCUMENT --"

def generate_summary(blocks,doc_type):
    summary={"DOCUMENT" :doc_type,
             "Father name":None,
             "DOB":None,
             "Number":None,
             "Issuing Authority":None,
             "Other detail":[]
             }
    for text in blocks:
        clean=text.strip()
        upper=clean.upper()

        if "ELECTION COMMISION OF INDIA" in upper:
            summary["Issuing Authority"]="Election Commission of India"
        if "GOVERNMENT OF INDIA" in upper:
            summary["Issuing Authority"]="Government of india"
    return summary  

def process_document(image_path):
    image=cv2.imread(image_path)
    if image is None:
        return {"error" : f"could not read your image : {image_path}"}
    blocks,image=extract_text_blocks(image,draw_boxes=True)
    doc_type=detect_document_type(blocks)
    summary=generate_summary(blocks,doc_type)

    output_data={
        "filename":os.path.basename(image_path),
        "raw_cleaned_text":blocks,
        "cleaned_summary":summary,
        "document_type":doc_type
    }
    ts=time.strftime("%Y%m%d_%H%M%S")
    base_time=os.path.splitext(os.path.basename(image_path))[0]
    json_filename=os.path.join(OUTPUT_FOLDER,f"{base_time}_{ts}.json")
    with open(json_filename,"w",
              encoding="utf-8")as f:
        json.dump(output_data,f,ensure_ascii=False,indent=4)
    result_image_path=os.path.join(OUTPUT_FOLDER,f"{base_time}_{ts}_output.jpg")
    cv2.imwrite(result_image_path,image)
    return output_data

@app.route('/')
def index():
    return "<h1>home page</h1>"

@app.route('/process',methods=['GET'])
def process():
    image_name="C:\\Users\\ASUS\\Music\\xbiz-trial\\document_type_detection\\a2.jpg"
    image_path=os.path.join(os.getcwd(),image_name)
    result=process_document(image_path)
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True)

