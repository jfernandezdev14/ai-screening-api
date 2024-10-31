from fastapi import File, UploadFile
import pandas as pd
import numpy as np
import pickle


class MachineLearningService:
    def __init__(self, pickle_path=None, model_name="nearest_neighbors"):
        self.pickle_path = pickle_path
        self.model_name = model_name
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
    
    def extract_files_information(self, files: List[UploadFile] = File(...)):
        return self.path
    
    def load_model(self):
        self.model = pickle.loads(open(self.pickle_path, "rb"))
        return