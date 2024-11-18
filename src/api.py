from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import pandas as pd
from src.main import process_new_pdf, query_vectorstore
import io
import shutil

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
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")



# Ensure the uploads directory exists
os.makedirs("src/uploads", exist_ok=True)



retriever = None
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global retriever
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("./uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"./uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF
        retriever = await process_new_pdf(
            fpath="./uploads/",
            fname=file.filename
        )
        
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(request: dict):
    global retriever
    try:
        result = await query_vectorstore(request['question'], retriever)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_context/")
async def get_context(request: Request):
    data = await request.json()
    metadata = data.get("metadata")
    
    if not metadata or not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="Invalid metadata format")
    
    sources = metadata.get("sources")
    ranked_metadata = metadata.get("ranked_metadata")
    
    if not sources or not ranked_metadata:
        raise HTTPException(status_code=400, detail="Missing sources or ranked_metadata")

    highlights = []
    for element in ranked_metadata:
        for ele in element:
            filename = ele.get("filename")
            page_number = ele.get("pagenumber")
            bbox = ele.get("coordinates")
            layout_width = ele.get("layout_width")
            layout_height = ele.get("layout_height")    
            
            if not filename or page_number is None or not bbox:
                raise HTTPException(status_code=400, detail="Missing required metadata fields")

            # Update the path check to match the mounted directory
            pdf_path = os.path.join("./uploads", filename)
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
        
            highlights.append({
                "filename": filename,  # Just the filename, not the full path
                "page_number": page_number,
                "bbox": bbox,
                "layout_width": layout_width,
                "layout_height": layout_height
            })

    return {"highlights": highlights}

@app.post("/multi_query/")
async def multi_query(request: Request):
    data = await request.json()
    questions = data.get("questions", [])
    if not questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    results = []
    for question in questions:
        try:
            result = await query_vectorstore(question, retriever)
            results.append({
                "question": question,
                "result": result
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e)
            })
    
    return {"results": results}

@app.post("/upload_questions/")
async def upload_questions(file: UploadFile):
    try:
        # Read file content
        content = await file.read()
        
        # Get file extension
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in ['csv', 'xlsx']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Please upload CSV or XLSX file"
            )

        # Process file based on extension
        try:
            if file_extension == 'csv':
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            else:  # xlsx
                df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}"
            )

        # Validate columns
        if 'question' not in df.columns and 'questions' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="File must contain a 'question' or 'questions' column"
            )

        # Get the question column
        question_col = 'question' if 'question' in df.columns else 'questions'
        
        # Extract and validate questions
        questions = df[question_col].dropna().str.strip().tolist()
        
        # Remove empty questions and duplicates
        questions = list(filter(None, questions))
        questions = list(dict.fromkeys(questions))  # Remove duplicates while preserving order

        if not questions:
            raise HTTPException(
                status_code=400,
                detail="No valid questions found in file"
            )

        return {
            "status": "success",
            "questions": questions,
            "message": f"Successfully loaded {len(questions)} questions"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)