# Use the official lightweight Python 3.9 Debian-based slim image to minimize container image size
FROM python:3.9-slim

# Set the active working directory inside the container filesystem to /app
WORKDIR /app

# Copy all files from the local build context (backend_files directory) into the container's working directory
COPY . .

# Install pinned Python dependencies without caching wheels/tarballs to keep the image layer footprint minimal
RUN pip install --no-cache-dir -r requirements.txt

# Document that the Flask container listens on port 7860
EXPOSE 7860

# Launch the Gunicorn production WSGI server with 4 concurrent worker processes bound to port 7860
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
