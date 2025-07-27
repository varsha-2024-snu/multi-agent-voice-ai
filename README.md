# Multi-Agent Voice Conversational System

This project is a voice-based conversational assistant featuring two distinct AI agents — an **Optimist** and a **Realist**. It allows users to interact via speech and receive responses from both agents, offering dual perspectives on any topic, imitating a group discussion.

Built using:
- LangGraph for multi-agent workflow
- Google Gemini (via LangChain) for response generation
- Deepgram for speech-to-text and text-to-speech
- Chroma for memory storage
- PyDub and ffmpeg for audio playback
- Python with a minimal local setup

---

## Features

- Voice-based interaction with two AI personalities
- Real-time transcription and text-to-speech
- Agents provide both optimistic and realistic views
- Modular design using LangGraph for stateful flow
- Offline-friendly components with only minimal online calls for LLM and TTS

---

### Future Enhancements

- **GUI Interface:** Build a lightweight graphical interface to display agent responses and conversation history.
- **Offline Capability:** Integrate offline speech recognition and TTS models to reduce dependency on internet access.
- **Multilingual Support:** Add ability to interact in Indian regional languages like Hindi, Tamil, Malayalam, etc.
- **Agent Personality Customization:** Allow users to choose or configure agent personas (e.g., humorous, academic, mentor).
- **Conversation Logging:** Maintain logs of past interactions for analytics or feedback.
- **Interrupt and Replay Controls:** Add hotkeys or voice commands to interrupt, repeat, or skip agent responses.
- **Emotion Detection:** Integrate basic sentiment analysis or emotion detection to adapt response tone.
- **Mobile/Web App Integration:** Extend the system to work on smartphones or browsers via a client-server architecture.



## Setup Instructions

### Clone the repository

```bash
git clone https://github.com/your-username/multi_agent_voice_conv_sys.git
cd multi_agent_voice_conv_sys

### Step 1: Create a virtual environment using Python 3.11

python3.11 -m venv .venv

### Step 2: Activate the virtual environment

source .venv/bin/activate

### Step 3: Install all Python packages

pip install -r requirements.txt



