import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

# Load Job Description
with open("job_description.txt", "r", encoding='utf-8') as file:
    job_description = file.read()

# Function to extract text from PDF
def extract_resume_text(file_path):
    try:
        return extract_text(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Load all resume texts
resume_dir = "resumes"
resume_texts = []
resume_names = []

for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(resume_dir, filename)
        text = extract_resume_text(path)
        resume_texts.append(text)
        resume_names.append(filename)

# Combine job description + resumes
documents = [job_description] + resume_texts

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity
job_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]
similarities = cosine_similarity(job_vector, resume_vectors).flatten()

# Rank resumes
ranking = pd.DataFrame({
    "Resume": resume_names,
    "Similarity Score": similarities
}).sort_values(by="Similarity Score", ascending=False)

print("Top Matching Resumes:\n")
print(ranking.to_string(index=False))
