FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for PyQt and OpenCV/matplotlib bindings)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libfontconfig1 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-image0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-xfixes0 \
    libxcb-sync1 \
    libxcb-util1 \
    libdbus-1-3 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libasound2 \
    libxrandr2 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose API port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Default command to run the headless FastAPI backend
CMD ["uvicorn", "codes.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
