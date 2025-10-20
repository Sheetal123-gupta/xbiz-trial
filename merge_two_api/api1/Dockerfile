# Use Python 3.11
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy code
COPY . /app

# Install dependencies
RUN pip install flask requests

# Run Flask app
CMD ["python", "app.py"]
