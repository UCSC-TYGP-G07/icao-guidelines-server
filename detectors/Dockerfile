# Base image from Dockerhub with a minimal version of Python
FROM python:slim

# To ensure that Python outputs its standard output immediately
ENV PYTHONUNBUFFERED 1

# Install build-essential which includes CMake and other build tools
RUN apt-get update && apt-get install -y build-essential python3-opencv

# To reduce the size of the Docker image by removing the package lists
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose port 8000 in the container (where the application will run)
EXPOSE 8000

# Make /app as a working directory in the container
WORKDIR /app

# Copy requirements from host, to docker container in /app
COPY ./requirements.txt .

# Copy everything from ./src directory to /app in the container
COPY ./utilities/ ./utilities/
COPY ./blur/laplacian/ ./blur/laplacian/
COPY ./eyes/ ./eyes/
COPY ./face/ ./face/
COPY ./geometric_tests/geometric_tests.py ./geometric_tests/geometric_tests.py
COPY ./varied_background/grab_cut_mean.py ./varied_background/grab_cut_mean.py

COPY server.py icao_photo_validator.py ./

# Install the specified dependencies using pip
RUN pip install -r requirements.txt

# Run the application in port 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "server:app"]
