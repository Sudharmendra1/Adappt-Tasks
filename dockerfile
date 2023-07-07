

# Use a base image with the desired operating system and runtime
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /Users/sudharmendragv/Downloads/Adappt

# Copy the ML model files to the working directory
#COPY /Users/sudharmendragv/Downloads/Adappt .
COPY requirements.txt .


# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose any necessary ports (if applicable)
#EXPOSE 8000

# Define the command to run your ML model
CMD ["python", "employee.py"]
