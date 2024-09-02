from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import pandas as pd
from src.main import process_new_pdf, query_vectorstore

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve the static files (index.html)
app.mount("/static", StaticFiles(directory="src/static", html=True), name="static")

# Serve the uploads directory
app.mount("/uploads", StaticFiles(directory="src/uploads"), name="uploads")

# Path to the CSV file for storing mappings
csv_file_path = "src/uploads/mapping.csv"

# Ensure the uploads directory exists
os.makedirs("src/uploads", exist_ok=True)

# Initialize the CSV file if it doesn't exist
if not os.path.exists(csv_file_path):
    df = pd.DataFrame(columns=["original_filename", "id_filename", "file_size"])
    df.to_csv(csv_file_path, index=False)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    # Read the file content to calculate its size
    file_content = await file.read()
    file_size = len(file_content)
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Check for duplicate files
    duplicate = df[(df["original_filename"] == file.filename) & (df["file_size"] == file_size)]
    if not duplicate.empty:
        return {"filename": file.filename, 
                "message":"Duplicate detected. Using the cached Document",
                "status":"processed", 
                "id_filename": duplicate.iloc[0]["id_filename"]}
    
    # Determine the new ID for the file
    id_number = len(df) + 1
    id_filename = f"{id_number:04d}.pdf"
    
    # Save the uploaded file with the new ID
    file_path = os.path.join("src/uploads", id_filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    # Save the mapping to the DataFrame and then to the CSV file
    new_row = {"original_filename": file.filename, "id_filename": id_filename, "file_size": file_size}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file_path, index=False)
    
    # Process the new PDF
    await process_new_pdf("src/uploads/", id_filename)
    
    return {"filename": file.filename, 
            "id_filename": id_filename, 
            "status": "processed", 
            "message": "Upload Successful"}

@app.get("/query/")
async def query_pdf(query: str):
    try:
        result = await query_vectorstore(query)
        return {"query": query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_context/")
async def get_context(request: Request):
    data = await request.json()
    metadata = data.get("metadata")
    if not metadata or not isinstance(metadata, list) or not metadata[0]:
        raise HTTPException(status_code=400, detail="Invalid metadata format")

    highlights = []
    for element in metadata:
        for ele in element:
            filename = ele.get("filename")
            page_number = ele.get("pagenumber")
            bbox = ele.get("coordinates")
            layout_width = ele.get("layout_width")
            layout_height = ele.get("layout_height")    
            if not filename or page_number is None or not bbox:
                raise HTTPException(status_code=400, detail="Missing required metadata fields")

            pdf_path = os.path.join("src/uploads", filename)
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=404, detail="PDF file not found")

            
            highlights.append({"filename": filename, "page_number": page_number, "bbox": bbox, "layout_width": layout_width, "layout_height": layout_height})

    return {"highlights": highlights}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)