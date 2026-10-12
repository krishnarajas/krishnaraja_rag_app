# Step 1: Use an official lightweight Python runtime
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies (Required for building ChromaDB's dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your Python script and data files into the container
COPY . .

# Step 6: Define the command to run your app
# (Assuming your script is named app.py)
CMD ["python", "RAG_app_gemini.py"]