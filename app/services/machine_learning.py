import os
from typing import List
import uuid
from fastapi import File, UploadFile
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

from app.config import settings
import pandas as pd
import numpy as np
import fitz
import re
import spacy
import pickle



class MachineLearningService:
    def __init__(self, pickle_path=None, model_name="random_forest"):
        self.model_selected = model_name
        self.pickle_path = pickle_path
        self.nlp_model = spacy.load('en_core_web_md')
        self.uploaded_file_paths = []
        self.models_mapping = {}
        self.tfidf_vectorizer = None
        
        nn_model_path = self.pickle_path + 'neural_network_model.pkl'
        rf_model_path = self.pickle_path + 'random_forest_model.pkl'
        knn_model_path = self.pickle_path + 'knn_model.pkl'
        tfidf_vectorizer_path = self.pickle_path + 'tfidf_vectorizer.pkl'
        cleaned_df_path = self.pickle_path + 'data_cleaned.pkl'
        
        if pickle_path: 
            # Load the trained models and vectorizer
            if Path(nn_model_path).is_file():
                with open(nn_model_path, 'rb') as file:
                    loaded_nn_model = pickle.load(file)

            if Path(rf_model_path).is_file():
                with open(self.pickle_path +'random_forest_model.pkl', 'rb') as file:
                    loaded_rf_model = pickle.load(file)

            if Path(knn_model_path).is_file():
                with open(self.pickle_path + 'knn_model.pkl', 'rb') as file:
                    loaded_knn_model = pickle.load(file)

            self.models_mapping = {
                "neural_network": loaded_nn_model,
                "random_forest": loaded_rf_model,
                "nearest_neighbors": loaded_knn_model
            }

            # Load the TF-IDF vectorizer (assuming it was saved)
            if Path(tfidf_vectorizer_path).is_file():
                with open(self.pickle_path +'tfidf_vectorizer.pkl', 'rb') as file:
                    tfidf_vectorizer = pickle.load(file)
            
                self.tfidf_vectorizer = tfidf_vectorizer

            # Load the cleaned DataFrame (assuming it was saved)
            if Path(cleaned_df_path).is_file():
                cleaned_df = pd.read_pickle(self.pickle_path +'data_cleaned.pkl')
                self.cleaned_df = cleaned_df

    def load_model(self):
        self.data_ingestion_process(self.pickle_path)

    def data_ingestion_process(self, path):
        
        # Load the data from the XLSX file
        df = pd.read_excel('./db/data.xlsx', index_col=0)
        
        # Preprocess the data
        df = self.preprocess_dataframe(df)
        
        # Compute TF-IDF vectors
        vectorizer, job_title_vectors, profile_vectors = self.compute_tfidf_vectors(df)

        # Save the TF-IDF vectorizer to a pickle file
        self.save_tfidf_vectorizer(vectorizer)

        # Save the cleaned DataFrame
        self.save_cleaned_dataframe(df)

        # Train and save the models
        self.train_and_save_models(job_title_vectors, profile_vectors)

        return

    def train_and_save_models(self, job_title_vectors, profile_vectors):

        #Creating X and y for training
        X= np.concatenate([profile_vectors.toarray(), job_title_vectors.toarray()], axis=1)
        y= 1 - self.cleaned_df['disqualified']

        #Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Neural Network (MLP Regressor)
        nn_model = self.train_neural_network(X_train, X_test, y_train, y_test)
        self.models_mapping["neural_network"] = nn_model

        #Random Forest Regressor
        rf_model = self.train_random_forest_model(X_train, X_test, y_train, y_test)
        self.models_mapping["random_forest"] = rf_model

        #K-Nearest Neighbors Regressor
        knn_model = self.train_knn_model(X_train, X_test, y_train, y_test)
        self.models_mapping["nearest_neighbors"] = knn_model

        # Save the models
        self.save_trained_models(nn_model, rf_model, knn_model)

    def save_cleaned_dataframe(self, df):
        
        # Save the cleaned DataFrame to a pickle file
        df_clean = df[['job title', 'headline', 'summary', 'keywords', 'educations', 'experiences', 'skills', 'candidate_profile', 'disqualified']]
        df_clean.to_pickle('./db/data_cleaned.pkl')
        self.cleaned_df = df_clean

    def save_tfidf_vectorizer(self, vectorizer):

        with open('./db/tfidf_vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
        
        self.tfidf_vectorizer = vectorizer


    def calculate_similarity_scores(self, df, job_title_vectors, profile_vectors):
        similarity_scores = cosine_similarity(job_title_vectors, profile_vectors)

        # Step 5: Extract the diagonal (similarity of each job title with its corresponding candidate profile)
        df['score'] = np.diag(similarity_scores)

    def compute_tfidf_vectors(self, df):
        
        #Create the TFIDF vectorizer with a maximum of 200 features
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')

        # Combine job_title and candidate_profile into one list of documents for vectorization
        all_text = df['job title'].tolist() + df['candidate_profile'].tolist()

        # Step 3: Fit and transform text data
        tfidf_matrix = vectorizer.fit_transform(all_text)

        # Separate vectors for job titles and candidate profiles
        job_title_vectors = tfidf_matrix[:len(df)]
        profile_vectors = tfidf_matrix[len(df):]

        return vectorizer,job_title_vectors,profile_vectors

    def preprocess_dataframe(self, df):

        #Strip the whitespaces from the data
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        #Strip the whitespaces from the column names
        df.columns = df.columns.str.strip()

        #Replace unwanted values with nan (empty)
        df = df.replace(["N/A"], np.nan)

        #Adjust the column names to lowercase
        df.columns = df.columns.str.lower()

        #Drop rows with missing values
        df = df[['job title', 'headline', 'summary', 'keywords', 'educations','experiences', 'skills', 'disqualified']]

        #Map the values of the Disqualified column to 1 and 0
        order = {"Yes": 1, "No": 0}
        df["disqualified"] = df["disqualified"].map(order)

        #Remove noise from columns with object data type
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col]= df[col].str.replace(r"_x[0-9a-fA-F]{4}_", "", regex=True)

        #Fill missing values with empty string
        df['summary'] = df['summary'].str.lower().fillna('')

        str_cols = ['headline', 'summary','educations', 'experiences', 'skills']
        for col in df.columns:
            if col in str_cols:
                df[col]= df[col].str.replace(r"_x[0-9a-fA-F]{4}_", "", regex=True).astype(str)

        #Combine the text columns into one column
        df['candidate_profile'] = df[['headline', 'summary','educations', 'experiences', 'skills']].apply(lambda x: ' '.join(x), axis=1)

        return df

    def save_trained_models(self, nn_model, rf_model, knn_model):

        # Save trained models in db folder
        with open(self.pickle_path + 'neural_network_model.pkl', 'wb') as file:
            pickle.dump(nn_model, file)

        with open(self.pickle_path + 'random_forest_model.pkl', 'wb') as file:
            pickle.dump(rf_model, file)

        with open(self.pickle_path + 'knn_model.pkl', 'wb') as file:
            pickle.dump(knn_model, file)

    def train_knn_model(self, X_train, X_test, y_train, y_test):

        # Tune the hyperparameters
        knn_model = self.tune_knn_hyperparameters(X_train, y_train)

        # Train the model
        knn_model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred_knn = knn_model.predict(X_test)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        print(f"KNN MSE on Test Set: {mse_knn}")
        return knn_model

    def tune_knn_hyperparameters(self, X_train, y_train):

        # Define the parameter grid for K-Nearest Neighbors
        param_grid_knn = {
            'n_neighbors': [3, 5, 7, 10],                 # Number of neighbors
            'weights': ['uniform', 'distance'],           # Weighting function
            'p': [1, 2]                                   # Distance metric (1 for Manhattan, 2 for Euclidean)
        }

        # K-Nearest Neighbors Regressor
        knn_model = KNeighborsRegressor()

        # Instantiate GridSearchCV for tunning model
        grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

        # Fit GridSearchCV
        grid_search_knn.fit(X_train, y_train)

        # Output best parameters and best MSE
        print("Best KNN Parameters:", grid_search_knn.best_params_)
        print("Best KNN MSE:", -grid_search_knn.best_score_)

        # Output best parameters
        best_knn_params = grid_search_knn.best_params_
        print("Best KNN Parameters:", best_knn_params)

        # Instantiate the model with the best parameters
        best_knn_model = KNeighborsRegressor(**best_knn_params)

        return best_knn_model

    def train_random_forest_model(self, X_train, X_test, y_train, y_test):

        # Tune the hyperparameters
        rf_model = self.tune_random_forest_model(X_train, y_train)

        # Train the model
        rf_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred_rf = rf_model.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        print(f"Random Forest MSE on Test Set: {mse_rf}")

        return rf_model

    def tune_random_forest_model(self, X_train, y_train):

        # Define the parameter grid for Random Forest
        param_grid_rf = {
            'n_estimators': [50, 100, 200],           # Number of trees
            'max_depth': [None, 10, 20, 30],          # Maximum depth of each tree
            'min_samples_split': [2, 5, 10],          # Minimum samples required to split a node
            'min_samples_leaf': [1, 2, 4],            # Minimum samples required at each leaf node
            'max_features': [None, 'sqrt', 'log2']  # Number of features to consider for each split
        }

        # Random Forest Regressor
        rf_model = RandomForestRegressor(random_state=42)

        # Instantiate GridSearchCV for tunning model
        grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

        # Fit GridSearchCV
        grid_search_rf.fit(X_train, y_train)

        # Output best parameters and best MSE
        print("Best Random Forest Parameters:", grid_search_rf.best_params_)
        print("Best Random Forest MSE:", -grid_search_rf.best_score_)

        # Output best parameters
        best_rf_params = grid_search_rf.best_params_
        print("Best Random Forest Parameters:", best_rf_params)

        # Instantiate the model with the best parameters
        best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
        return best_rf_model

    def train_neural_network(self, X_train, X_test, y_train, y_test):

        # Tune the hyperparameters
        nn_model = self.tune_neural_network_hyperparameters(X_train, y_train)

        # Train the model
        nn_model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred_nn = nn_model.predict(X_test)
        mse_nn = mean_squared_error(y_test, y_pred_nn)

        print(f"Neural Network MSE on Test Set: {mse_nn}")

        return nn_model

    def tune_neural_network_hyperparameters(self, X_train, y_train):
        
        # Define the parameter grid for Neural Network
        param_grid_nn = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different layer configurations
            'activation': ['relu', 'tanh'],                             # Activation functions
            'learning_rate_init': [0.001, 0.01, 0.1],                   # Learning rates
            'alpha': [0.0001, 0.001, 0.01],                             # L2 regularization (alpha)
            'max_iter': [200, 300]                                      # Maximum number of iterations
        }

        # Neural Network (MLP Regressor)
        nn_model = MLPRegressor(random_state=42)

        # Instantiate GridSearchCV for tunning
        grid_search_nn = GridSearchCV(nn_model, param_grid_nn, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

        # Fit GridSearchCV
        grid_search_nn.fit(X_train, y_train)

        # Output best parameters and best MSE
        print("Best Neural Network Parameters:", grid_search_nn.best_params_)
        print("Best Neural Network MSE:", -grid_search_nn.best_score_)

        # Output best parameters
        best_nn_params = grid_search_nn.best_params_
        print("Best Neural Network Parameters:", best_nn_params)

        # Instantiate the model with the best parameters
        best_nn_model = MLPRegressor(**best_nn_params, random_state=42)
        return best_nn_model

    def predict(self, input_data):

        # Analyze and get best model
        # best_model, _ = self._predict_best_model(input_data)

        # Get related rows
        related_rows = self._get_related_rows(input_data, self.cleaned_df, self.tfidf_vectorizer, self.model_selected)

        result = related_rows.to_dict()

        return result
    
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
            self.uploaded_file_paths.append(file_location)

        return cleaned_text
    

    def _extract_text_from_pdf(self,pdf_file):

        text = []
        with fitz.open(pdf_file) as pdf:
            for page_num in range(0, pdf.page_count):
                page = pdf[page_num]
                text.append(page.get_text().replace("\n", " ").lower())
            
        full_text = " ".join(text)
            
        return full_text
        
    def _clean_text(self, text):
        doc = self.nlp_model(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)
    
    def _vectorize_text(self, text):
        # Transform the input text using the TF-IDF vectorizer
        return self.tfidf_vectorizer.transform([text])

    def _predict_best_model(self,input_text):

        # Transform the input text using the TF-IDF vectorizer
        text_vector = self._vectorize_text(input_text)

        # Make predictions with each model
        nn_prediction = self.models_mapping["neural_network"].predict(text_vector)
        rf_prediction = self.models_mapping["random_forest"].predict(text_vector)
        knn_prediction =  self.models_mapping["nearest_neighbors"].predict(text_vector)

        # Store predictions in a dictionary
        predictions = {
            "neural_network": nn_prediction[0],
            "random_forest": rf_prediction[0],
            "nearest_neighbors": knn_prediction[0]
        }

        # Find the model with the best prediction
        best_model = max(predictions, key=predictions.get)  # Get the model with the highest score
        best_prediction = predictions[best_model]

        print(f"Best Model: {best_model}")
        print(f"Prediction Score: {best_prediction}")

        return self.models_mapping[best_model], best_prediction
    

    def _get_related_rows(self, input_text, cleaned_df, vectorizer, model):
        
        # Transform the input text using the TF-IDF vectorizer
        input_vector = self._vectorize_text(input_text)

        # Calculate similarity scores for each row in the DataFrame
        candidate_profile_vectors = vectorizer.transform(cleaned_df['candidate_profile'])

        all_vectors = np.array([np.concatenate([candidate_profile_vectors.toarray()[i] , input_vector.toarray()[0]]) for i in range(len(candidate_profile_vectors.toarray()))])

        predictions = model.predict(all_vectors)

        similarity_scores = np.dot(candidate_profile_vectors, input_vector.T).toarray()

        # Normalize prediction scores to a scale of 0 to 100
        min_score = 0
        max_score = predictions.max()
        
        # Avoid division by zero if max_score is equal to min_score
        if max_score != min_score:
            normalized_scores = (predictions - min_score) / (max_score - min_score) * 100
        else:
            normalized_scores = np.zeros_like(predictions)  # All scores are the same, set to 0%

        # Add similarity scores to the DataFrame
        cleaned_df['similarity_score'] = similarity_scores

        # Store normalized scores as a new column in the DataFrame
        cleaned_df['normalized_prediction_score'] = normalized_scores

        # Combine predictions and similarity scores in the DataFrame
        cleaned_df['prediction'] = predictions

        # Sort by similarity score and return the top 30 rows
        top_related_rows = cleaned_df.sort_values(by='normalized_prediction_score', ascending=False).head(30)

        return top_related_rows

