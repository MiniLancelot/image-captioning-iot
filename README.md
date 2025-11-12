# Image Captioning IoT

A simple image-captioning project for IoT experiments. This repository contains code and resources to run a minimal image captioning demo locally or on an edge device.

## Features
- Simple demo to generate captions for images
- Minimal dependencies and quick setup
- Intended for experimentation and teaching

## Prerequisites
- Git
- Python 3.8+ (or your preferred Python 3.x)
- pip

## Environment (Gemini API)
If you plan to use the Gemini API, set an environment variable for your API key (example name: GEMINI_API_KEY). This repo expects the key to be available in the environment when running code that calls Gemini.

See the Gemini API documentation for authentication and usage details (search "Gemini API" or consult your provider's docs).

## Installation
1. Clone the repository:
   ```
   git clone <repo-url> D:\DUT\Year5\IOT\image-captioning-iot
   cd D:\DUT\Year5\IOT\image-captioning-iot
   ```

2. Create and activate a virtual environment:
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - Windows (cmd):
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. Install Python dependencies (if requirements.txt exists):
   ```
   pip install -r requirements.txt
   ```
   If there is no requirements file, install packages used by the project as needed.

## Quick run (example)
- If the repo has a main script:
  ```
  python main.py
  ```
- If it's a Flask/FastAPI app:
  ```
  # Example for Flask
  export FLASK_APP=app.py
  flask run
  ```

Replace the commands above with the actual entry point of this project.

## Project structure (example)
- README.md
- requirements.txt
- app.py / main.py
- models/           # stored or downloaded models

## Models and .gitignore
To avoid committing large model files, add the models folder to your .gitignore. Example entry for .gitignore:
```gitignore
# Ignore downloaded or exported model files
models/
```

Place downloaded model/voice files under the repository's models/ directory (create subfolders as needed). If you want to keep an empty directory tracked, add a small placeholder like models/.gitkeep and do not ignore that file.

For prebuilt voice models and examples, see the Rhasspy Piper voices repository on Hugging Face:
https://huggingface.co/rhasspy/piper-voices/tree/main

Download the assets you need from that page and copy them into models/ (for example: models/piper-voices/<voice-name>/).
