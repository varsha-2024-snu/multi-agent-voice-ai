import os
import getpass
from uuid import uuid4
from typing import TypedDict, List
import requests
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph

from stt_tts import record_and_trans, text_to_speech

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(
    collection_name="conversation_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.7
)

class AgentState(TypedDict):
    messages: List

def store_message(role: str, content: str):
    doc = Document(page_content=content, metadata={"role": role})
    uid = str(uuid4())
    vector_store.add_documents([doc], ids=[uid])

def retrieve_context(query: str, k: int = 3) -> str:
    results = vector_store.similarity_search(query, k=k)
    return "\n".join(
        f"{r.metadata['role'].capitalize()}: {r.page_content}"
        for r in results
    )

def search_serper(query):
    headers = {"X-API-KEY": os.environ["SERPER_API_KEY"]}
    payload = {"q": query}
    res = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
    return res.json().get("organic", [])[:3]

def get_latest_news(topic):
    params = {
        "q": topic,
        "apiKey": os.environ["NEWS_API_KEY"],
        "language": "en",
        "pageSize": 3,
    }
    res = requests.get("https://newsapi.org/v2/everything", params=params)
    return res.json().get("articles", [])[:3]

def optimist_agent(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content
    store_message("user", user_input)

    search_results = search_serper(user_input)
    news_articles = get_latest_news(user_input)

    search_summary = "\n".join(f"- {r['title']} ({r['link']})" for r in search_results)
    news_summary = "\n".join(f"- {a['title']} ({a['url']})" for a in news_articles)
    past_context = retrieve_context(user_input)

    prompt = (
        "You are an optimistic advisor. Highlight exciting opportunities.\n"
        f"User said: {user_input}\n"
        f"Relevant past conversation:\n{past_context}\n\n"
        f"Top Search Results:\n{search_summary}\n\n"
        f"Related News Articles:\n{news_summary}\n\n"
        "Now write a positive and informative reply using this information IN ONLY 100 WORDS MAX."
    )

    response = llm.invoke(prompt).content
    store_message("optimist", response)
    print(f"\nOptimist Agent: {response}")
    text_to_speech(response, agent="optimist")
    return {"messages": [AIMessage(content=response)]}

def realist_agent(state: AgentState) -> AgentState:
    user_input = state["messages"][-1].content
    store_message("optimist", user_input)

    search_results = search_serper(user_input)
    news_articles = get_latest_news(user_input)

    search_summary = "\n".join(f"- {r['title']} ({r['link']})" for r in search_results)
    news_summary = "\n".join(f"- {a['title']} ({a['url']})" for a in news_articles)
    past_context = retrieve_context(user_input)

    prompt = (
        "You are a realist advisor. Give practical advice.\n"
        f"User query: {user_input}\n"
        f"Relevant past conversation:\n{past_context}\n\n"
        f"Web Search Context:\n{search_summary}\n"
        f"Related News Articles:\n{news_summary}\n\n"
        "Now provide a realistic perspective in ONLY 100 WORDS MAX and ask a follow-up question."
    )

    response = llm.invoke(prompt).content
    store_message("realist", response)
    print(f"\nRealist Agent: {response}")
    text_to_speech(response, agent="realist")
    return {"messages": [AIMessage(content=response)]}

def handoff_to_user(state: AgentState) -> AgentState:
    print("\nyour reply: ")
    user_reply = record_and_trans()
    if user_reply.strip().lower() == "exit":
        print("Conversation ended by user.")
        exit(0)
    store_message("user", user_reply)
    return {"messages": [HumanMessage(content=user_reply)]}

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

print("Start the conversation by speaking...")
first_input = record_and_trans()
initial_input = {"messages": [HumanMessage(content=first_input)]}

final_state = graph.invoke(initial_input)

for msg in final_state["messages"]:
    print(f"{msg.type.capitalize()}: {msg.content}")
