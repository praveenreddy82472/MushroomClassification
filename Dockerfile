# Use Python 3.9 slim version as the base image
FROM python:3.9-slim

# Install necessary packages
RUN apt-get update -y && apt-get install -y awscli && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the contents of the current directory to /app inside the container
COPY . /app

# Upgrade pip, setuptools, and wheel to the latest version
RUN pip install --upgrade pip setuptools wheel

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install python-multipart to handle form data in FastAPI
RUN pip install python-multipart

# Run the main application
CMD ["python3", "main.py"]
