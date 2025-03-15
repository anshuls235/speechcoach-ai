# SpeechCoach AI

An AI-powered speech training assistant for practicing and improving your communication skills through instant feedback.

## Features

- **Three Training Modules**:
  - **Impromptu Speaking**: Practice thinking and speaking on your feet with structured, spontaneous responses
  - **Storytelling**: Develop engaging narrative skills with feedback on structure, emotion, and delivery
  - **Conflict Resolution**: Learn diplomatic communication techniques for handling difficult conversations

- **Multi-Modal Interaction**:
  - Practice through text input or voice recording
  - Get AI coaching feedback through text and audio responses
  - Speech-to-text transcription for natural practice sessions

- **Session Management**:
  - Create and save multiple training sessions
  - Switch between different modes and practice sessions
  - Track your progress over time

## Technical Details

- Built with Llama 3.1 8B (default) for optimal performance and 128K context window
- Uses Whisper for accurate speech recognition
- Employs TTS (Text-to-Speech) for natural-sounding audio feedback
- Developed with Gradio for an intuitive, responsive interface

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speechcoach-ai.git
   cd speechcoach-ai
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   python app.py
   ```

### Using Different Models

You can specify a different language model using an environment variable:

```bash
# Use Phi-4
export LLM_MODEL="microsoft/phi-4"
python app.py

# Use Mistral 7B
export LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
python app.py
```

## Usage Guide

1. **Create a Session**: Click "Create New Session" and select your preferred training module
2. **Practice**: Type your response or use the microphone to record your speech
3. **Receive Feedback**: Get detailed AI coaching on your performance
4. **Listen to Responses**: Click on the assistant's messages to hear them spoken aloud
5. **Track Progress**: Use multiple sessions to practice different skills over time

## Deployment

### Docker

Build and run with Docker:
```bash
docker build -t speechcoach-ai .
docker run -p 7860:7860 speechcoach-ai
```

### Hugging Face Spaces

This application can be deployed on Hugging Face Spaces:
1. Create a new Gradio Space
2. Upload the application files
3. Configure the hardware (GPU recommended for better performance)

## Requirements

- Python 3.8+
- CUDA-enabled GPU recommended for optimal performance
- Internet connection for model downloads (first run)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with Hugging Face Transformers
- Uses TTS and Whisper for audio processing
- Powered by Gradio for the user interface

---

*SpeechCoach AI: Perfect your communication skills through AI-powered practice and feedback.*