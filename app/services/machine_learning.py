import os
from typing import List
import uuid
from fastapi import File, UploadFile
from pathlib import Path
from app.config import settings
import pandas as pd
import numpy as np
import fitz
import re
import spacy
import pickle



class MachineLearningService:
    def __init__(self, pickle_path=None, model_name="nearest_neighbors"):
        self.pickle_path = pickle_path
        self.model_name = model_name
        self.nlp_model = spacy.load("en_core_web_sm")
        self.uploaded_file_paths = []
        if self.pickle_path:
            self.load_model()

    def load(self, path):
        
        dfx = pd.read_excel("./db/data.xlsx", index_col=0)
        
        dfx.info()

        # Step 1: Strip the whitespaces from the data
        dfx = dfx.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Step 2: Strip the whitespaces from the column names
        dfx.columns = dfx.columns.str.strip()

        # Step 3: Replace unwanted values with nan (empty)
        dfx = dfx.replace(["N/A"], np.nan)

        # Step 4: Turn boolean values to 1 and 0
        order = {"Yes": 1, "No": 0}
        dfx["Disqualified"] = dfx["Disqualified"].map(order)

        



        dfx.info()
        print(dfx.isnull().sum()/len(dfx)*100)
        return

    def predict(self, data):
        self.model.predict(data)
        return 

    def train(self, data, target):
        return self.model.fit(data, target)

    def evaluate(self, data, target):
        return self.model.score(data, target)
    
    async def process_data_files(self, files: List[UploadFile] = File(...)):

        for file in files:
            
            file_extension = Path(file.filename).suffix
            if file_extension != ".pdf":
                raise ValueError("Only PDF files are allowed.")
            
            unique_id = uuid.uuid4()
            new_file_name = str(unique_id) + "_candidate_data.pdf"
            file_location = os.path.join(settings.FILES_DIRECTORY, new_file_name)

            with open(file_location, "wb") as f:
                f.write(await file.read())
            
            text = self._extract_text_from_pdf(file_location)
            cleaned_text = self._clean_text(text)
            candidate_features = self._extract_candidate_features(cleaned_text)
            self.uploaded_file_paths.append(file_location)

        return
    
    def load_model(self):
        self.model = pickle.loads(open(self.pickle_path, "rb"))
        return
    

    def _extract_text_from_pdf(self,pdf_file):

        text = []
        with fitz.open(pdf_file) as pdf:
            for page_num in range(0, pdf.page_count):
                page = pdf[page_num]
                text.append(page.get_text().replace("\n", " ").lower())
            
        full_text = " ".join(text)
            
        return full_text
        
    
    # Data Ingestion and Preprocessing
    def _clean_text(self, text):
        doc = self.nlp_model(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

    # Feature Extraction: Extract relevant candidate features
    def _extract_candidate_features(self, text):
        doc = self.nlp_model(text)
        skills = list(set(re.findall(r'\b(Golang|Ruby on Rails|PostgreSQL|JavaScript|TypeScript)\b', text, re.I)))
        
        experience = re.search(r'(\d+)\+?\s+years', text)
        education = re.search(r'(BSc|Bachelor\'s|Master\'s)\s+(in\s+[A-Za-z\s]+)', text)
        certifications = re.findall(r'(Certified|Certification|Certificate)', text, re.I)


        return {
            "Skills": " ".join(skills),
            "Experience": experience.group(0) if experience else "0 years",
            "Education": education.group(0) if education else "N/A",
            "Certifications": " ".join(certifications) if certifications else "N/A",
        }
