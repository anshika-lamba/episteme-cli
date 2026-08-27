# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install the required dependencies
RUN pip install --no-cache-dir pexpect openai pydantic

# Copy the necessary Python scripts into the container
COPY mini_episteme.py atif.py run_paired_trial.py epsilon_gate.py ./

# Define the entrypoint to run the MVP script
ENTRYPOINT ["python", "mini_episteme.py"]

    
