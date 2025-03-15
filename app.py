import os
import json
import uuid
import copy
import time
from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
import gradio as gr
import sqlite3
from datetime import datetime
from transcription import transcribe
from tts import generate_speech
from huggingface_hub import login

# Get token from environment variable (set via Space secrets)
token = os.environ.get("HF_TOKEN")

# Login if token is available
if token:
    login(token) 


#Load the LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BNB_CONFIG, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Global variable to track current session
current_session = None

# Session management
SESSIONS_FILE = "sessions.json"
def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, 'r') as f:
            return json.load(f)
    return {"sessions": {}}


def save_sessions(sessions_data):
    """Save sessions data to a JSON file, handling non-serializable objects."""
    # Create a copy to avoid modifying the original
    sessions_copy = copy.deepcopy(sessions_data)

    # Process each session
    for session_id, session in sessions_copy["sessions"].items():
        if "history" in session:
            # Process history to make it JSON serializable
            serializable_history = []
            for msg in session["history"]:
                # Convert ChatMessage objects to dict
                if isinstance(msg, gr.ChatMessage):
                    serializable_history.append({
                        "role": msg.role,
                        "content": msg.content if isinstance(msg.content, str) else msg.metadata["text"]
                    })
                else:
                    serializable_history.append(msg)

            # Replace original history with serializable version
            session["history"] = serializable_history

    # Save to file
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions_copy, f, indent=2)

def user(user_message, history, current_session_name, sessions_data):
    if not user_message.strip():
        return "", history, sessions_data, gr.Radio(interactive=True)

    current_session = next(
        (session for session in sessions_data["sessions"].values()
         if session["name"] == current_session_name),
        None
    )

    if not current_session:
        print(f"Session not found: {current_session_name}")
        return user_message, history, sessions_data, gr.Radio(interactive=True)

    history.append({"role": "user", "content": user_message})
    current_session["history"] = history

    radio_interactive = True
    if len(history) == 1:
        current_session["mode_locked"] = True
        radio_interactive = False
        print(f"Locking mode for session: {current_session_name}")

    save_sessions(sessions_data)

    return "", history, sessions_data, gr.Radio(interactive=radio_interactive)


def predict(current_session_name, history, sessions_data):
    # Find session by name
    current_session = None
    for sid, session in sessions_data["sessions"].items():
        if session["name"] == current_session_name:
            current_session = session
            break

    if not current_session:
        return history, sessions_data

    # Check if history is empty
    if not history:
        print("Error: History is empty")
        return history, sessions_data

    current_mode = current_session["mode"]

    prompt_template = [{'role': 'system', 'content': d_system_prompts[current_mode]}] + history
    input_text = tokenizer.apply_chat_template(prompt_template, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.7,
        "streamer": streamer
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    history.append({"role": "assistant", "content": ""})
    for text in streamer:
        cleaned_text = text.replace("assistant\n\n", "")
        history[-1]['content'] += cleaned_text

        # Update session history
        current_session["history"] = history
        save_sessions(sessions_data)

        yield history, sessions_data, None


def create_new_session(mode_selection, sessions_data):
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session_name = f"{mode_selection} - {timestamp}"
    update_current_session(session_name)

    # Create a new session
    sessions_data["sessions"][session_id] = {
        "id": session_id,
        "name": session_name,
        "mode": mode_selection,
        "mode_locked": False,
        "created_at": timestamp,
        "history": []
    }

    # Save sessions data
    save_sessions(sessions_data)

    # Update session dropdown
    session_choices = [(s["name"], s["name"]) for s in sessions_data["sessions"].values()]

    return (
        gr.Dropdown(choices=session_choices, value=session_name),
        sessions_data,
        gr.Column(visible=True),
        gr.Radio(value=mode_selection, interactive=False if sessions_data["sessions"][session_id]["mode_locked"] else True),
        gr.Chatbot(value=sessions_data["sessions"][session_id]["history"], visible=True),
        gr.Textbox(visible=True, value=""),
        gr.Button(visible=True, interactive=False),
        gr.Audio(visible=False, interactive=False),
        gr.Markdown(visible=True)
    )

def session_changed(session_name, sessions_data):

    update_current_session(session_name)
    # Find session by name
    selected_session = None
    session_id = None

    for sid, session in sessions_data["sessions"].items():
        if session["name"] == session_name:
            selected_session = session
            session_id = sid
            break

    if not selected_session:
        # explicitly hide all components including audio_output
        return (
            gr.Column(visible=False),
            gr.Radio(interactive=True, visible=False),
            gr.Chatbot(value=[], visible=False),
            gr.Textbox(visible=False),
            gr.Audio(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            gr.Markdown(visible=False)
        )

    mode_locked = selected_session["mode_locked"]

    if selected_session["mode"] == 'Impromptu Speaking':
        placeholder = "Practice impromptu speaking skills"
    elif selected_session["mode"] == 'Storytelling':
        placeholder = "Develop storytelling abilities"
    else:
        placeholder = "Learn conflict resolution techniques"

    # Explicitly set audio_output visibility to True when loading existing session
    return (
        gr.Column(visible=True),
        gr.Radio(value=selected_session["mode"], interactive=not mode_locked, visible=True),
        gr.Chatbot(value=selected_session["history"], visible=True),
        gr.Textbox(placeholder=placeholder, visible=True, value=""),
        gr.Audio(visible=True, interactive=True, value=None),
        gr.Button(visible=True, interactive=False),
        gr.Audio(visible=False, interactive=False, value=None),
        gr.Markdown(visible=True)
    )


def update_mode(mode_selection, current_session_name, sessions_data):
    # Find session by name
    current_session = None
    current_session_id = None

    for sid, session in sessions_data["sessions"].items():
        if session["name"] == current_session_name:
            current_session = session
            current_session_id = sid
            break

    if not current_session:
        return sessions_data, gr.Dropdown()

    # Update mode only if it's not locked
    if not current_session["mode_locked"]:
        old_name = current_session["name"]
        current_session["mode"] = mode_selection

        # Update the session name
        timestamp = current_session["created_at"]
        new_name = f"{mode_selection} - {timestamp}"
        current_session["name"] = new_name

        # Save sessions data
        save_sessions(sessions_data)

        # Update session dropdown
        session_choices = [(s["name"], s["name"]) for s in sessions_data["sessions"].values()]
        return sessions_data, gr.Dropdown(choices=session_choices, value=new_name)

    return sessions_data, gr.Dropdown()

def change_button(text_input):
    if text_input:
        return gr.Button(interactive=True)
    else:
        return gr.Button(interactive=False)

def setup_feedback_db():
    """Create the SQLite database and table if they don't exist"""
    db_path = 'feedback.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS message_feedback (
        session_id TEXT,
        message_index INTEGER,
        liked INTEGER,
        feedback_time TIMESTAMP,
        message_content TEXT,
        PRIMARY KEY (session_id, message_index)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Feedback database initialized at {db_path}")
    return db_path

def update_current_session(session_name):
    global current_session
    current_session = session_name
    print(f"Current session updated to: {current_session}")

def like(evt: gr.LikeData):
    """
    Save user feedback to SQLite database
    
    Args:
        evt: Gradio LikeData event containing index, liked status, and value
    """
    try:
        # Use the global current session variable
        global current_session
        session_id = current_session or "unknown_session"
        
        print(f"User {'liked' if evt.liked else 'unliked'} the response at index {evt.index}")
        print(f"Session ID: {session_id}")
        
        # Connect to database
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        # Message content (truncate if too long)
        message_content = json.dumps(evt.value)[:1000] if evt.value else ""
        
        # Check if record already exists
        cursor.execute(
            "SELECT * FROM message_feedback WHERE session_id = ? AND message_index = ?",
            (session_id, evt.index)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute(
                "UPDATE message_feedback SET liked = ?, feedback_time = ? WHERE session_id = ? AND message_index = ?",
                (1 if evt.liked else 0, timestamp, session_id, evt.index)
            )
            print(f"Updated feedback for session {session_id}, message {evt.index}")
        else:
            # Insert new record
            cursor.execute(
                "INSERT INTO message_feedback (session_id, message_index, liked, feedback_time, message_content) VALUES (?, ?, ?, ?, ?)",
                (session_id, evt.index, 1 if evt.liked else 0, timestamp, message_content)
            )
            print(f"Saved new feedback for session {session_id}, message {evt.index}")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error saving feedback to database: {str(e)}")

def listen(evt: gr.SelectData):
  if evt.index % 2 == 1:
    audio = gr.Audio(value=generate_speech(evt.value), autoplay=True)
    return audio
  else:
    return None
  

#Gradio App Code
db_path = setup_feedback_db()
with gr.Blocks(theme="soft",css="""
    footer {display: none !important;}
    /* Increased max-width and set width percentage */
    .gradio-container {max-width: 1400px !important; margin: 0 auto; width: 95% !important;}
    .gr-button {border-radius: 8px !important;}
    .gr-box {border-radius: 10px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;}
    /* Force full width on Gradio container elements */
    #component-0 {width: 100% !important;}
    #component-0 > div {width: 100% !important;}
""") as demo:
    # Add professional HTML header
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 1rem; background: linear-gradient(90deg, #3a7bd5, #2d65b9); padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: white; font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 600;">SpeechCoach AI</h1>
            <p style="color: #e0e0e0; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.5;">
                Your personal AI-powered speech training assistant. Practice anywhere, anytime, and receive instant feedback to perfect your communication skills.
            </p>
        </div>
    """)
    
    # Initialize sessions data and get default session
    initial_sessions = load_sessions()
    sessions_data = gr.State(initial_sessions)
    
    # Determine default session if available
    default_session = None
    if initial_sessions["sessions"] and len(initial_sessions["sessions"]) > 0:
        # Get the first session as default
        default_session = list(initial_sessions["sessions"].values())[0]["name"]
    
    # Session management
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            session_dropdown = gr.Dropdown(
                label="Select Session",
                choices=[(s["name"], s["name"]) for s in initial_sessions["sessions"].values()] if initial_sessions["sessions"] else [],
                allow_custom_value=False,
                value=default_session  # Set the default value
            )
        with gr.Column(scale=1):
            new_session_btn = gr.Button("Create New Session")
            
    # Get initial visibility state based on whether there are existing sessions
    initial_visibility = default_session is not None
    
    # Mode selection (initially hidden until a session is created)
    with gr.Column(visible=initial_visibility) as mode_column:
        # If there's a default session, get its mode
        default_mode = None
        if default_session is not None and default_session in [s["name"] for s in initial_sessions["sessions"].values()]:
            for session in initial_sessions["sessions"].values():
                if session["name"] == default_session:
                    default_mode = session.get("mode", "Impromptu Speaking")
                    break
        
        radio = gr.Radio(
            choices=["Impromptu Speaking", "Storytelling", "Conflict Resolution"],
            label="Training Module",
            info="Select mode for your new session",
            value=default_mode or "Impromptu Speaking"
        )
        
    # Chatbox and inputs - set visibility based on default session
    chatbox = gr.Chatbot(
        type="messages", 
        show_copy_button=True, 
        height=400, 
        visible=initial_visibility,
        avatar_images=("user.webp","agent.webp"),
        label="🔊 AI Chatbot: Tap Assistant's Reply to Hear It!",
        show_label=True
    )

    help_markdown = gr.Markdown("""
  ## 📝 How to Use  
✨ **Type your prompt** in the textbox **OR**  
🎤 **Record/Upload audio** – it will be transcribed automatically!  

👉 **Don't forget to click "Send"** for the chatbot to process your message!
  """, visible=False)
    
    with gr.Row(equal_height=True):
        audio_input = gr.Audio(
            label="Speak/Upload your response", 
            sources=["upload", "microphone"], 
            visible=initial_visibility
        )
        text_input = gr.Textbox(
            label="Enter your response:", 
            visible=initial_visibility
        )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            audio_output = gr.Audio(visible=initial_visibility)
        with gr.Column(scale=1):
            submit = gr.Button("Send", visible=initial_visibility)
    
    # Create new session
    new_session_btn.click(
        create_new_session,
        [radio, sessions_data],
        [session_dropdown, sessions_data, mode_column, radio, chatbox, text_input, submit, audio_output, help_markdown]
    )
    
    # Session selection handling
    session_dropdown.change(
        session_changed,
        [session_dropdown, sessions_data],
        [mode_column, radio, chatbox, text_input, audio_input, submit, audio_output, help_markdown]
    )
    
    # Mode update handling
    radio.change(
        update_mode,
        [radio, session_dropdown, sessions_data],
        [sessions_data, session_dropdown]
    )
    
    # Audio transcription
    audio_input.change(transcribe, audio_input, [text_input, submit])
    
    # Text input handling
    text_input.change(change_button, text_input, submit)
    
    # Message submission
    submit.click(
        user,
        [text_input, chatbox, session_dropdown, sessions_data],
        [text_input, chatbox, sessions_data, radio],
        queue=False
    ).then(
        predict,
        [session_dropdown, chatbox, sessions_data],
        [chatbox, sessions_data, audio_output]
    ).then(
        lambda: None,
        None,
        audio_input
    )
    
    # Listen to assistant's reply
    chatbox.select(listen, None, audio_output)
    
    # Feedback
    chatbox.like(like)
    
    # Load default session on app initialization if one exists
    if default_session is not None:
        demo.load(
            fn=session_changed,
            inputs=[session_dropdown, sessions_data],
            outputs=[mode_column, radio, chatbox, text_input, audio_input, submit, audio_output, help_markdown]
        )

demo.launch(debug=True)

