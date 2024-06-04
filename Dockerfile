# Use the Paketo Miniconda base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Install lib deps
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Create a new conda environment named 'fea' with Python 3.9
RUN conda create --name fea python=3.9 -y

# Activate the 'fea' environment and install dependencies
RUN /bin/bash -c "source activate fea && pip install --upgrade pip && pip install -r requirements.txt"

# Copy the rest of the application code to the working directory
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Activate the 'fea' environment and run the Flask application
CMD ["/bin/bash", "-c", "source activate fea && python app.py"]

