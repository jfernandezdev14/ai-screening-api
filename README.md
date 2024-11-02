# ai-screening-api

AI-Powered Candidate Screening and Scoring System

This repository contains a FastAPI application designed to provide a RESTful API for an AI-Powered Candidate Screening and Scoring System.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [License](#license)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following installed on your machine:

- **Python 3.8+**
- **pip** (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jfernandezdev14/ai-screening-api.git
   cd ai-screening-api
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies:**

   Make sure the requirements.txt file is in the root directory of the project, then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Running the Application**

   To start the FastAPI server, run the following command in the terminal:

   ```bash
   uvicorn main:app --reload
   ```

   - main is the name of the Python file where your FastAPI app instance is located (e.g., app = FastAPI()).
   - app is the instance name.

   The app will now be running on http://127.0.0.1:8000.

5. **API Documentation**

   FastAPI provides automatic interactive API documentation, available at:

   - Swagger UI: http://127.0.0.1:8000/docs
