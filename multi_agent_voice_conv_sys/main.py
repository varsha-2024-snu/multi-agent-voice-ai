import os
from uuid import uuid4
from typing import TypedDict, List, Dict
import requests
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph
from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions
import asyncio
import io
import simpleaudio as sa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize services with API keys from .env
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(
    collection_name="conversation_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Deepgram client
dg_config = DeepgramClientOptions(api_key=os.getenv("DEEPGRAM_API_KEY"))
dg_client = DeepgramClient(dg_config)

# Constants
AUDIO_FILE = 'input.wav'
VOICE = 'aura-asteria-en'

async def transcribe() -> str:
    """Transcribe audio file using Deepgram"""
    try:
        with open(AUDIO_FILE, 'rb') as audio:
            buffer_data = audio.read()
        
        options = PrerecordedOptions(
            model="nova-2",
            punctuate=True,
        )
        
        response = await dg_client.listen.prerecorded.v("1").transcribe_file(
            buffer_data,
            options
        )
        return response.results.channels[0].alternatives[0].transcript
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""

def text_to_speech(text: str):
    """Convert text to speech using Deepgram"""
    try:
        response = requests.post(
            f'https://api.deepgram.com/v1/speak?model={VOICE}',
            headers={
                'Authorization': f'Token {os.getenv("DEEPGRAM_API_KEY")}',
                'Content-Type': 'application/json'
            },
            json={'text': text}
        )
        
        if response.status_code == 200:
            audio_bytes = io.BytesIO(response.content)
            wave_obj = sa.WaveObject.from_wave_file(audio_bytes)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error in text-to-speech: {e}")


class AgentState(TypedDict):
    messages: List

def store_message(role: str, content: str):
    """Store conversation in vector database"""
    doc = Document(page_content=content, metadata={"role": role})
    uid = str(uuid4())
    vector_store.add_documents([doc], ids=[uid])
    vector_store.persist()

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant conversation history"""
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([f"{r.metadata['role'].capitalize()}: {r.page_content}" for r in results])

def search_serper(query: str) -> List[Dict]:
    """Search the web using Serper API"""
    headers = {"X-API-KEY": os.getenv("SERPER_API_KEY")}
    payload = {"q": query}
    res = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
    return res.json().get("organic", [])[:3]

def get_latest_news(topic: str) -> List[Dict]:
    """Get news articles using NewsAPI"""
    params = {
        "q": topic,
        "apiKey": os.getenv("NEWS_API_KEY"),
        "language": "en",
        "pageSize": 3,
    }
    res = requests.get("https://newsapi.org/v2/everything", params=params)
    return res.json().get("articles", [])[:3]

def optimist_agent(state: AgentState) -> AgentState:
    """Generate optimistic response"""
    user_input = state["messages"][-1].content
    store_message("user", user_input)

    # Gather context
    search_results = search_serper(user_input)
    news_articles = get_latest_news(user_input)
    past_context = retrieve_context(user_input)

    # Format context
    search_summary = "\n".join([f"- {r['title']} ({r['link']})" for r in search_results])
    news_summary = "\n".join([f"- {a['title']} ({a['url']})" for a in news_articles])

    prompt = f"""You are an optimistic advisor. Highlight exciting opportunities.
User said: {user_input}
Relevant past conversation:
{past_context}

Top Search Results:
{search_summary}

Related News Articles:
{news_summary}

Now write a positive and informative reply using this information in 200 words."""

    response = llm.invoke(prompt).content
    store_message("optimist", response)
    print(f"\n\U0001F7E2 Optimist Agent: {response}")
    text_to_speech(response)
    return {"messages": [AIMessage(content=response)]}

def realist_agent(state: AgentState) -> AgentState:
    """Generate realistic response"""
    user_input = state["messages"][-1].content
    store_message("optimist", user_input)

    # Gather context
    search_results = search_serper(user_input)
    news_articles = get_latest_news(user_input)
    past_context = retrieve_context(user_input)

    # Format context
    search_summary = "\n".join([f"- {r['title']} ({r['link']})" for r in search_results])
    news_summary = "\n".join([f"- {a['title']} ({a['url']})" for a in news_articles])

    prompt = f"""You are a realist advisor. Give practical advice.
User query: {user_input}
Relevant past conversation:
{past_context}

Web Search Context:
{search_summary}

Related News Articles:
{news_summary}

Now provide a realistic perspective in 200 words and ask a follow-up question."""

    response = llm.invoke(prompt).content
    store_message("realist", response)
    print(f"\n\U0001F535 Realist Agent: {response}")
    text_to_speech(response)
    return {"messages": [AIMessage(content=response)]}

def handoff_to_user(state: AgentState) -> AgentState:
    """Get user input from audio file"""
    user_reply = asyncio.run(transcribe())
    print(f"\n\U0001F9D1 You (from audio): {user_reply}")
    store_message("user", user_reply)
    return {"messages": [HumanMessage(content=user_reply)]}

# Build and run the conversation graph
def main():
    builder = StateGraph(AgentState)
    builder.add_node("Optimist Agent", optimist_agent)
    builder.add_node("Realist Agent", realist_agent)
    builder.add_node("User Input", handoff_to_user)

    builder.set_entry_point("Optimist Agent")
    builder.add_edge("Optimist Agent", "Realist Agent")
    builder.add_edge("Realist Agent", "User Input")
    builder.add_edge("User Input", "Optimist Agent")
    builder.set_finish_point("User Input")

    graph = builder.compile()

    # Start conversation with audio input
    initial_text = asyncio.run(transcribe())
    if not initial_text:
        print("Failed to transcribe initial audio input")
        return

    print(f"\n\U0001F9D1 Initial input (from audio): {initial_text}")

    final_state = graph.invoke({
        "messages": [HumanMessage(content=initial_text)]
    })

    # Print final conversation
    for msg in final_state["messages"]:
        print(f"{msg.type.capitalize()}: {msg.content}")

if __name__ == "__main__":
    main()