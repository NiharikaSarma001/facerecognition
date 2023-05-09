#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
from deepface import DeepFace
import numpy as np

app = FastAPI()

# Create an empty database to store the registered faces
face_db = {}

@app.post('/register_face')
async def register_face_endpoint(
    name: str,
    image_file: bytes,
):
    # Load image
    npimg = np.frombuffer(image_file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect face
    detector = DeepFace.build_model("MTCNN")
    detected_faces = detector.detectFace(img)

    # Extract features
    recognizer = DeepFace.build_model("Facenet")
    features = recognizer.extractFeatures(detected_faces)

    # Save to database
    face_db[name] = features

    return JSONResponse({'message': 'Face registered successfully'})

@app.post('/recognize_face')
async def recognize_face_endpoint(
    image_file: bytes,
    threshold: Optional[float] = 0.5,
):
    # Load image
    npimg = np.frombuffer(image_file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect face
    detector = DeepFace.build_model("MTCNN")
    detected_faces = detector.detectFace(img)

    # Extract features
    recognizer = DeepFace.build_model("Facenet")
    query = recognizer.extractFeatures(detected_faces)

    # Find closest match
    min_distance = float('inf')
    closest_match = None
    for name, features in face_db.items():
        distance = DeepFace.distance(query, features)
        if distance < min_distance:
            min_distance = distance
            closest_match = name

    if closest_match is None or min_distance > threshold:
        return JSONResponse({'message': 'No match found'})

    return JSONResponse({'name': closest_match, 'distance': min_distance})

