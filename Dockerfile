# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Expose the port the app runs on
EXPOSE 80

# Define environment variable for FastAPI's host
ENV HOST 0.0.0.0
ENV PORT 80

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]