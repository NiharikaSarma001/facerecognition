from deepface import DeepFace
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse

app = FastAPI()

# create a DataFrame to store registered students' information
df = pd.DataFrame(columns=['Name', 'Embedding'])

@app.post("/register_face")
async def register_face(name: str, file: UploadFile = File(...)):
    # save the image locally
    file_location = f"registered_faces/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # load image
    img = Image.open(file_location)
    
    # convert PIL image to numpy array
    img_array = np.array(img)

    # generate face embedding
    embedding = DeepFace.represent(img_array, enforce_detection=False)

    # add the name and embedding to the DataFrame
    df.loc[len(df)] = [name, embedding]

    # return success message
    return {"message": f"{name}'s face has been registered."}


@app.post("/recognize_face")
async def recognize_face(file: UploadFile = File(...)):
    # save the image locally
    file_location = f"input_faces/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # load image
    img = Image.open(file_location)
    
    # convert PIL image to numpy array
    img_array = np.array(img)

    # generate face embedding
    input_embedding = DeepFace.represent(img_array, enforce_detection=False)

    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # extract the embedding from the row
        embedding = np.array(row['Embedding'][0]['embedding'])
        
        # compute the cosine similarity between the embeddings
        similarity = 1 - cosine(input_embedding[0]['embedding'], embedding)
        
        # check if similarity score is above a threshold
        if similarity > 0.7:
            # return success message with name of the recognized person
            return {"message": f"Welcome {row['Name']}!"}

    # return error message if no match is found
    return JSONResponse(status_code=400, content={"message": "Sorry, face not recognized."})

