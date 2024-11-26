# Use an official Python runtime as a base image
FROM python:3.12.4

# Set the working directory
WORKDIR /Visualisations-Recomendations-Draco-LLMs

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot's source code
COPY . .

# Set the command to run the bot
CMD ["python", "main.py"]
