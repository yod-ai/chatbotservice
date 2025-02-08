import os
import json
import csv
import re
import unicodedata
from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime
from uuid import uuid4
from rapidfuzz import fuzz  # Improved fuzzy matching via RapidFuzz
from pydantic import BaseModel
from dotenv import load_dotenv
import wikipedia

# --- Monkey-patch BeautifulSoup to use lxml by default ---
import bs4

original_bs4_init = bs4.BeautifulSoup.__init__


def new_bs4_init(self, *args, **kwargs):
    if "features" not in kwargs:
        kwargs["features"] = "lxml"
    original_bs4_init(self, *args, **kwargs)


bs4.BeautifulSoup.__init__ = new_bs4_init

# --- Updated LangChain & LangGraph Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph

# --- Updated SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship
from sqlalchemy.sql import func

# --- Load environment variables ---
load_dotenv()


# --- Helper Functions ---
def sanitize_query(query: str) -> Optional[str]:
    """
    Scans the user input and rejects it if any banned phrases (e.g. "search the web", "google", etc.) are found.
    """
    banned_phrases = ["search the web", "google", "bing", "web search", "scrape"]
    lower_query = query.lower()
    for phrase in banned_phrases:
        if phrase in lower_query:
            return None
    return query


def refine_response(text: str) -> str:
    """
    Removes common markdown artifacts (such as '*' and '#' characters) from the text.
    """
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalizes text using NFKC normalization, making it more robust for fuzzy matching.
    """
    return unicodedata.normalize("NFKC", text).lower().strip()


# --- Configuration Class ---
class Config:
    def __init__(self):
        self.LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'
        self.LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
        self.LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///chatbot.db')
        self.DEFAULT_RESPONSES_PATH = os.getenv('DEFAULT_RESPONSES_PATH', 'default_responses.json')
        self.MIN_DOCS_THRESHOLD = 2
        self.BREEDS_CSV_PATH = os.getenv('BREEDS_CSV_PATH', 'breeds.csv')


# --- SQLAlchemy ORM Setup ---
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    has_pet = Column(Boolean, default=False)
    pet_breed = Column(String, nullable=True)
    pet_type = Column(String, nullable=True)  # 'dog' or 'cat'
    chats = relationship("Chat", back_populates="user")


class Chat(Base):
    __tablename__ = 'chats'
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    session_id = Column(String, nullable=False)  # Unique session ID for each chat session
    timestamp = Column(DateTime, default=func.now())
    messages = Column(JSON)  # Stores conversation as a list of message dicts
    user = relationship("User", back_populates="chats")


class DefaultResponse(Base):
    __tablename__ = 'default_responses'
    id = Column(Integer, primary_key=True)
    pattern = Column(String, nullable=False)
    response = Column(String, nullable=False)


# --- Pydantic State Definition ---
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime


class PetAssistantState(TypedDict):
    messages: List[Message]
    user_id: str
    session_id: str
    query: str
    query_type: str
    retrieved_docs: List[str]
    current_response: str
    requires_human: bool
    reflection: Optional[str]
    confidence_score: float
    has_pet: bool
    pet_info: Optional[Dict]


# --- Main Chatbot Class ---
class PetAssistant:
    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-3.5-turbo",
            openai_api_key=self.config.OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings()
        self.engine = create_engine(self.config.DATABASE_URL, echo=False)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)
        self.db = self.Session()
        self.vectorstores = self.initialize_vectorstores()
        self.default_responses = self.load_default_responses()
        self.breed_data = self.load_breed_data()
        self.workflow = self.create_workflow()

    # --- Vectorstore Initialization with Persistence ---
    def _init_vectorstore(self, persist_dir: str, doc_path: str) -> Chroma:
        if os.path.exists(persist_dir):
            return Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        else:
            loader = PyPDFLoader(doc_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            vectorstore.persist()
            return vectorstore

    def initialize_vectorstores(self) -> Dict[str, Chroma]:
        return {
            "adoption": self._init_vectorstore("adoption_db", "adoption_docs.pdf"),
            "medical": self._init_vectorstore("medical_db", "medical_docs.pdf"),
            "training": self._init_vectorstore("training_db", "training_docs.pdf")
        }

    # --- Load Breed Data from CSV ---
    def load_breed_data(self) -> Dict[str, str]:
        breed_data = {}
        csv_path = self.config.BREEDS_CSV_PATH
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        breed = row["breed"].strip().lower()
                        description = row["description"].strip()
                        breed_data[breed] = description
            except Exception as e:
                print(f"Error reading breed CSV file: {e}")
        else:
            print(f"Warning: Breed CSV file '{csv_path}' not found.")
        return breed_data

    # --- Load Default Responses ---
    def load_default_responses(self) -> Dict[str, str]:
        try:
            with open(self.config.DEFAULT_RESPONSES_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "çalışma saatleriniz nedir": "Hafta içi 09:00 - 18:00, Cumartesi 10:00 - 16:00 arası hizmet veriyoruz.",
                "evlat edinme ücreti nedir": "Evlat edinme ücretlerimiz hayvana göre değişmektedir. Lütfen web sitemizi ziyaret edin veya bizimle iletişime geçin.",
                "acil servis var mı": "Acil veteriner hizmeti sunmuyoruz. Lütfen en yakın acil hayvan hastanesine başvurun.",
                "merhaba": "Merhaba! Size nasıl yardımcı olabilirim?",
                "nasılsınız": "İyiyim, teşekkür ederim. Siz nasılsınız?",
                "evcil hayvanım yardıma ihtiyacı var": "Elbette, evcil hayvanınızın sorunu hakkında detay verebilir misiniz?"
            }

    # --- Get and Update User Info ---
    def get_user_info(self, user_id: str) -> Dict:
        user = self.db.query(User).filter_by(id=user_id).first()
        if not user:
            user = User(id=user_id, has_pet=False)
            self.db.add(user)
            self.db.commit()
        return {
            "has_pet": user.has_pet,
            "pet_breed": user.pet_breed,
            "pet_type": user.pet_type
        }

    def update_user_pet_info(self, user_id: str, pet_type: str, pet_breed: Optional[str]):
        user = self.db.query(User).filter_by(id=user_id).first()
        if user:
            if pet_type.lower() == "sıfırla":  # Reset pet info command
                user.has_pet = False
                user.pet_type = None
                user.pet_breed = None
            else:
                user.has_pet = True
                user.pet_type = pet_type
                user.pet_breed = pet_breed
            self.db.commit()

    # --- Improved Fuzzy Matching for Default Responses ---
    def check_default_response(self, query: str) -> Optional[str]:
        query_clean = normalize_text(query)
        best_score = 0
        best_response = None
        for pattern, response in self.default_responses.items():
            pattern_clean = normalize_text(pattern)
            score = fuzz.token_set_ratio(query_clean, pattern_clean)
            if score > best_score:
                best_score = score
                best_response = response
        if best_score >= 80:
            return best_response
        return None

    # --- Retrieve Context Documents ---
    def retrieve_context(self, state: PetAssistantState) -> PetAssistantState:
        docs = []
        user_info = self.get_user_info(state["user_id"])
        # For queries not classified as "adoption" and if pet info exists, add breed info from CSV and Wikipedia.
        if state["query_type"] != "adoption" and user_info["has_pet"] and user_info["pet_breed"]:
            breed_lower = user_info["pet_breed"].strip().lower()
            csv_info = self.breed_data.get(breed_lower)
            if csv_info:
                docs.append(csv_info)
                wiki_breed_info = self.search_wikipedia(breed_lower)
                docs.extend(wiki_breed_info)
            else:
                wiki_breed_info = self.search_wikipedia(user_info["pet_breed"])
                docs.extend(wiki_breed_info)
        # Retrieve documents from the appropriate vectorstore.
        vectorstore = self.vectorstores.get(state["query_type"])
        if vectorstore:
            vs_docs = vectorstore.similarity_search(state["query"], k=3)
            docs.extend([doc.page_content for doc in vs_docs])
        # Supplement with Wikipedia if not enough documents.
        if len(docs) < self.config.MIN_DOCS_THRESHOLD:
            wiki_results = self.search_wikipedia(state["query"], pet_breed=user_info.get("pet_breed"))
            docs.extend(wiki_results)
        state["retrieved_docs"] = docs
        state["has_pet"] = user_info["has_pet"]
        state["pet_info"] = user_info if user_info["has_pet"] else None
        return state

    def search_wikipedia(self, query: str, pet_breed: Optional[str] = None) -> List[str]:
        try:
            search_query = f"{pet_breed} {query}" if pet_breed else query
            results = wikipedia.search(search_query, results=2)
            summaries = []
            for title in results:
                try:
                    page = wikipedia.page(title)
                    summaries.append(page.summary)
                except wikipedia.exceptions.DisambiguationError as e:
                    page = wikipedia.page(e.options[0])
                    summaries.append(page.summary)
                except Exception:
                    continue
            return summaries
        except Exception as e:
            print(f"Wikipedia arama hatası: {str(e)}")
            return []

    # --- Chat History and Session Management ---
    def store_chat(self, user_id: str, session_id: str, messages: List[Dict[str, Any]]):
        chat = Chat(
            id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            timestamp=datetime.now()
        )
        self.db.add(chat)
        self.db.commit()

    def get_chat_history(self, user_id: str, session_id: str) -> List[Dict]:
        chats = self.db.query(Chat).filter_by(user_id=user_id, session_id=session_id).all()
        return [{
            "id": chat.id,
            "timestamp": chat.timestamp.strftime("%d %B %Y %H:%M"),
            "messages": chat.messages
        } for chat in chats]

    def get_chat_sessions(self, user_id: str) -> List[Dict]:
        sessions = self.db.query(Chat.session_id, func.min(Chat.timestamp).label("first_ts")).filter_by(
            user_id=user_id).group_by(Chat.session_id).all()
        session_list = []
        for sess_id, first_ts in sessions:
            chats = self.get_chat_history(user_id, sess_id)
            if chats:
                first_msg = chats[0]["messages"][0]["content"]
                summary = " ".join(first_msg.split()[:3])
                session_list.append({
                    "session_id": sess_id,
                    "summary": summary,
                    "timestamp": first_ts.strftime("%d %B %Y %H:%M")
                })
        return session_list

    def get_contextual_summary(self, user_id: str, session_id: str) -> str:
        chats = self.get_chat_history(user_id, session_id)
        if not chats:
            return ""
        recent_chats = chats[-5:]
        conversation_text = ""
        for chat in recent_chats:
            for message in chat['messages']:
                conversation_text += f"{message['role']}: {message['content']}\n"
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Önceki konuşma geçmişini özetle, böylece yeni sorguya bağlam sağlayabilirsin:"),
            HumanMessage(content=conversation_text)
        ])
        formatted_summary_prompt = summary_prompt.format_messages()
        summary_response = self.llm.invoke(formatted_summary_prompt)
        return summary_response.content.strip()

    # --- Query Classification ---
    def classify_query(self, state: PetAssistantState) -> PetAssistantState:
        default_answer = self.check_default_response(state["query"])
        if default_answer:
            state["query_type"] = "default"
            state["current_response"] = default_answer
            state["confidence_score"] = 1.0
            state["reflection"] = "Varsayılan yanıt kullanıldı."
            return state

        classifier_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Aşağıdaki sorguyu şu kategorilerden birine sınıflandırın:
1. Evcil hayvan benimseme (adoption)
2. Evcil sağlık/tedavi (medical)
3. Evcil eğitim (training)
Lütfen JSON formatında, "category", "confidence" (0 ile 1 arasında) ve "reasoning" anahtarlarıyla cevap verin."""),
            HumanMessage(content=state["query"])
        ])
        formatted_prompt = classifier_prompt.format_messages(query=state["query"])
        response = self.llm.invoke(formatted_prompt)
        try:
            classification = json.loads(response.content)
        except Exception as e:
            classification = {"category": "default", "confidence": 0.0, "reasoning": f"Çözümleme hatası: {str(e)}"}
        allowed_categories = {"adoption", "medical", "training"}
        cat = classification.get("category") or ""
        if str(cat).lower() not in allowed_categories:
            state["query_type"] = "default"
            state[
                "current_response"] = "Lütfen daha net ifade edin ve yalnızca belirtilen evcil destek hizmetlerini kullanın!"
            state["confidence_score"] = 1.0
            state["reflection"] = "Girdi belirsiz veya geçersiz."
            return state
        state["query_type"] = str(cat).lower()
        state["confidence_score"] = classification.get("confidence", 0.0)
        state["reflection"] = classification.get("reasoning", "")
        return state

    # --- Response Generation ---
    def generate_response(self, state: PetAssistantState) -> PetAssistantState:
        if state["query_type"] == "default":
            self.store_chat(state["user_id"], state["session_id"], [
                {"role": "kullanıcı", "content": state["query"], "timestamp": datetime.now().isoformat()},
                {"role": "asistan", "content": state["current_response"], "timestamp": datetime.now().isoformat()}
            ])
            return state

        conversation_summary = self.get_contextual_summary(state["user_id"], state["session_id"])
        context = "\n".join(state["retrieved_docs"])
        if conversation_summary:
            context = f"Önceki konuşma özeti:\n{conversation_summary}\n\n" + context

        pet_context_medical = ""
        pet_context_training = ""
        if state["query_type"] != "adoption" and state["has_pet"] and state.get("pet_info"):
            pet_type = state["pet_info"].get("pet_type", "")
            pet_breed = state["pet_info"].get("pet_breed", "")
            pet_context_medical = f"Evcil hayvanınızın bir {pet_type} cinsi {pet_breed} olduğunu biliyorum. Buna göre size özel sağlık tavsiyeleri vereceğim."
            pet_context_training = f"Evcil hayvanınızın bir {pet_type} cinsi {pet_breed} olduğunu göz önünde bulundurarak, ona özel eğitim önerilerinde bulunacağım."
        system_prompts = {
            "medical": f"Sen bir evcil hayvan sağlık asistanısın. {pet_context_medical} Aşağıdaki bağlamı kullanarak sağlık bilgileri ver. Her zaman bir veterinerle görüşmeyi öneririm.",
            "adoption": "Sen bir evcil hayvan benimseme asistanısın. Sadece evlat edinme süreciyle ilgili bilgi ver.",
            "training": f"Sen bir evcil hayvan eğitim asistanısın. {pet_context_training} Aşağıdaki bağlamı kullanarak eğitim tavsiyeleri ver."
        }
        reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=f"{system_prompts.get(state['query_type'], '')}\n\nBu sorgunun insan müdahalesi gerektirip gerektirmediğini değerlendir."),
            HumanMessage(content=f"Bağlam: {context}\nSorgu: {state['query']}")
        ])
        formatted_reflection_prompt = reflection_prompt.format_messages(context=context, query=state["query"])
        reflection_response = self.llm.invoke(formatted_reflection_prompt)
        requires_human = "human intervention" in reflection_response.content.lower()

        response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"{system_prompts.get(state['query_type'], '')}\n\nBağlam: {context}"),
            HumanMessage(content=state["query"])
        ])
        formatted_response_prompt = response_prompt.format_messages()
        final_response = self.llm.invoke(formatted_response_prompt)
        refined_final_response = refine_response(final_response.content)

        self.store_chat(state["user_id"], state["session_id"], [
            {"role": "kullanıcı", "content": state["query"], "timestamp": datetime.now().isoformat()},
            {"role": "asistan", "content": refined_final_response, "timestamp": datetime.now().isoformat()}
        ])

        state["current_response"] = refined_final_response
        state["requires_human"] = requires_human
        state["reflection"] = reflection_response.content
        return state

    # --- Create Workflow ---
    def create_workflow(self):
        workflow = StateGraph(PetAssistantState)
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_response)
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.set_entry_point("classify")
        return workflow.compile()


# --- Testing Utilities & Main Entry ---
def create_test_assistant():
    os.environ['DATABASE_URL'] = 'sqlite:///chatbot.db'
    return PetAssistant()


def main():
    # Note: When integrating with the Flask backend, this interactive main() function is not used.
    assistant = PetAssistant()
    current_session_id = str(uuid4())
    print("Chatbot started. Type 'çık' to exit.")
    print(
        "Commands: 'geçmiş' (show chat sessions), 'evcil güncelle' (update pet info), 'evcil sil' (reset pet info), 'yeni sohbet' (start new chat).")
    user_id = str(uuid4())

    while True:
        user_input = input("\nSen: ").strip()
        lower_input = user_input.lower()
        if lower_input == 'çık':
            print("Teşekkürler, iyi günler!")
            break

        if lower_input == 'yeni sohbet':
            current_session_id = str(uuid4())
            print("Yeni sohbet başlatıldı.")
            continue

        if lower_input == 'geçmiş':
            sessions = assistant.get_chat_sessions(user_id)
            if sessions:
                print("\n--- Sohbet Geçmişi ---")
                for sess in sessions:
                    print(f"Oturum: {sess['summary']} - {sess['timestamp']} (Session ID: {sess['session_id']})")
            else:
                print("Henüz sohbet geçmişi yok.")
            continue

        if lower_input == 'evcil güncelle':
            pet_type = input("Evcil hayvan türü (dog/cat): ").strip().lower()
            if pet_type not in ['dog', 'cat']:
                print("Geçersiz tür. Sadece 'dog' veya 'cat' yazın.")
                continue
            pet_breed = input("Evcil hayvan cinsi: ").strip()
            assistant.update_user_pet_info(user_id, pet_type, pet_breed)
            print(f"Evcil bilgiler güncellendi: Tür: {pet_type}, Cins: {pet_breed}")
            continue

        if lower_input == 'evcil sil':
            assistant.update_user_pet_info(user_id, "sıfırla", None)
            print("Evcil bilgiler sıfırlandı.")
            continue

        sanitized = sanitize_query(user_input)
        if sanitized is None:
            print("Lütfen hizmeti yalnızca belirtilen amaçlar için kullanın!")
            continue

        try:
            state: PetAssistantState = {
                "messages": [],
                "user_id": user_id,
                "session_id": current_session_id,
                "query": sanitized,
                "query_type": "",
                "retrieved_docs": [],
                "current_response": "",
                "requires_human": False,
                "reflection": None,
                "confidence_score": 0.0,
                "has_pet": False,
                "pet_info": None
            }
            final_state = assistant.workflow.invoke(state)
            print(f"\nAsistan: {final_state['current_response']}")
            if final_state['requires_human']:
                print("\nNot: Bu sorgu insan müdahalesi gerektirebilir.")
            if final_state['confidence_score'] < 0.7:
                print("\nNot: Yanıt düşük güvenle verildi. Lütfen bir uzmana danışın.")
        except KeyboardInterrupt:
            print("\nSohbet kesildi. Hoşçakalın!")
            break
        except Exception as e:
            print(f"\nHata: {str(e)}. Lütfen sorgunuzu tekrar ifade edin.")
            continue


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Kritik hata: {str(e)}. Program sonlandırılıyor.")
