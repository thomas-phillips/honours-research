FROM python:3.8.10

RUN apt-get update && apt-get install -y libsndfile1 libsndfile1-dev ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .
