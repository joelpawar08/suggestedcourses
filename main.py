import os
import pdfplumber
import requests
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Groq client
groq_client = Groq(api_key="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")

# Set up YouTube API key
YOUTUBE_API_KEY = "AIzaSyCcC2iQOku8pBEsDEppFj3UjDsaiCakdW4"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


class CourseResponse(BaseModel):
    title: str
    platform: str
    link: str
    description: str
    duration: str
    level: str


def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text from uploaded PDF file"""
    with pdfplumber.open(pdf_file.file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text.strip() if text else "No text found in resume."


def analyze_skill_gap(resume_text: str, job_desc: str) -> str:
    """Use Groq AI to analyze skill gaps based on resume and job description"""
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI that analyzes skill gaps in resumes."},
            {"role": "user", "content": f"Resume:\n{resume_text}\n\nJob Description:\n{job_desc}\n\nIdentify missing skills."}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_completion_tokens=500,
        top_p=1
    )
    return response.choices[0].message.content.strip()


def get_youtube_courses(skill: str):
    """Search YouTube for relevant courses based on skill gaps and missing tech Stack"""
    params = {
        "part": "snippet",
        "q": f"{skill} online course",
        "key": YOUTUBE_API_KEY,
        "maxResults": 5
    }
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch YouTube courses")
    
    data = response.json()
    courses = []
    
    for item in data.get("items", []):
        video_id = item["id"].get("videoId")
        if not video_id:
            continue
        courses.append(CourseResponse(
            title=item["snippet"]["title"],
            platform="YouTube",
            link=f"https://www.youtube.com/watch?v={video_id}",
            description=item["snippet"]["description"],
            duration="Varies",
            level="Beginner to Advanced"
        ))
    
    return courses


@app.post("/suggest-courses")
async def suggest_courses(file: UploadFile = File(...), job_description: str = Form(...)):
    """Analyze skill gaps and return suggested courses"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Extract text from PDF
    resume_text = extract_text_from_pdf(file)

    # Analyze skill gaps
    skill_gaps = analyze_skill_gap(resume_text, job_description)
    missing_skills = [skill.strip() for skill in skill_gaps.split(",") if skill.strip()]

    # Get YouTube courses
    suggested_courses = []
    for skill in missing_skills:
        suggested_courses.extend(get_youtube_courses(skill))

    return {"courses": suggested_courses}
