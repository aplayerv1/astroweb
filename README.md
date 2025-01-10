# Real-time Signal Processing Web Interface

A Flask-based web application for real-time signal processing and visualization, featuring spectrograms and signal strength plots.

## Features

- Real-time signal processing and visualization
- Live spectrogram generation
- Signal strength monitoring
- Server-sent events for live logging
- REST API endpoints for data retrieval

## Technical Stack

- Flask web framework
- NumPy for signal processing
- Server-sent events (SSE) for real-time updates
- Thread-based continuous processing
- Queue-based logging system


## Setup

1. Clone the repository:
    - git clone this repository

2. Install dependencies:

    - pip install -r requirements.txt

3. Run the application:

    - python aic.py

## API Endpoints
    - Main web interface
        /api/plots - Get available plots and spectrograms
        /api/signal - Get latest signal processing data
        /logs - Stream real-time logs
