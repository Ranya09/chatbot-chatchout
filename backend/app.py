import os
import re
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
# S'assurer que pdf_indexer.py est dans le même répertoire ou PYTHONPATH
from pdf_indexer import PDFIndexer 
from fastapi import Query

# Configuration améliorée des logs
LOG_FILE_PATH = "app.log" # Fichier de log dans le répertoire courant
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Application démarrée et configuration du logging effectuée.")

# Load environment variables from .env file
load_dotenv()

# Obtenir la clé API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY non trouvée dans le fichier .env")
    raise ValueError("GROQ_API_KEY not found in .env file")
if not GROQ_MODEL:
    logger.error("GROQ_MODEL non trouvé ou vide dans le fichier .env")
    raise ValueError("GROQ_MODEL not found or empty in .env file")

logger.info(f"Clé API Groq et modèle chargés. Modèle utilisé: {GROQ_MODEL}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Client Groq initialisé avec succès.")
except Exception as e:
    logger.exception("Erreur lors de l'initialisation du client Groq.")
    raise

try:
    LEGAL_DOCS_FOLDER = "legal_documents"
    os.makedirs(LEGAL_DOCS_FOLDER, exist_ok=True)
    pdf_indexer = PDFIndexer(LEGAL_DOCS_FOLDER)
    logger.info(f"PDFIndexer initialisé pour le dossier: {LEGAL_DOCS_FOLDER}")
except Exception as e:
    logger.exception("Erreur lors de l'initialisation de PDFIndexer.")
    raise

class ResponseCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        logger.info(f"Cache de réponses initialisé avec une taille maximale de {max_size}.")
    
    def get(self, key):
        if key in self.cache:
            self.access_times[key] = time.time()
            logger.debug(f"Cache HIT pour la clé: {key}")
            return self.cache[key]
        logger.debug(f"Cache MISS pour la clé: {key}")
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            try:
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                logger.info(f"Cache plein. Élément le moins récent supprimé: {oldest_key}")
            except ValueError:
                logger.warning("Tentative de suppression d'un élément du cache alors que access_times est vide.")
        self.cache[key] = value
        self.access_times[key] = time.time()
        logger.debug(f"Clé mise en cache: {key}")
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache de réponses vidé.")

response_cache = ResponseCache(max_size=100)

class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

class FeedbackInput(BaseModel):
    conversation_id: str
    message_id: str
    rating: int
    comment: str = ""

class DocumentRequest(BaseModel):
    document_type: str
    language: str = "fr"
    parameters: dict

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": """Tu es un assistant juridique spécialisé dans le droit tunisien, capable de répondre en français et en arabe.

DIRECTIVES GÉNÉRALES :
1. Détecte automatiquement la langue de l'utilisateur (français ou arabe) et réponds dans la même langue
2. Justifie toujours tes réponses avec des références précises aux lois, codes et articles tunisiens
3. Structure tes réponses de manière claire et professionnelle
4. Si tu n'as pas d'information spécifique dans le contexte fourni, précise-le clairement
5. N'invente jamais de lois ou de dispositions légales qui ne sont pas mentionnées dans le contexte

FORMAT DE RÉPONSE EN FRANÇAIS :
- Commence par une réponse directe à la question
- Développe avec les détails juridiques pertinents
- Cite explicitement les articles de loi et références exactes (ex: \"Selon l'article 123 du Code du Travail tunisien...\")
- Termine par des recommandations pratiques ou des étapes à suivre

تنسيق الإجابة بالعربية:
- ابدأ بإجابة مباشرة على السؤال
- قم بتطوير إجابتك مع التفاصيل القانونية ذات الصلة
- استشهد صراحة بمواد القانون والمراجع الدقيقة (مثال: \"وفقًا للمادة 123 من مجلة الشغل التونسية...\")
- اختم بتوصيات عملية أو خطوات يجب اتباعها

Utilise les informations juridiques fournies dans le contexte pour répondre aux questions."""}
        ]
        self.active: bool = True
        self.last_activity: float = time.time()

    def update_last_activity(self):
        self.last_activity = time.time()

conversations: Dict[str, Conversation] = {}

def detect_language(text: str) -> str:
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    if arabic_pattern.search(text):
        return "arabic"
    return "french"

def query_groq_api(conversation: Conversation, user_query: str) -> str:
    logger.info(f"Début de query_groq_api pour la requête: {user_query[:50]}...")
    try:
        last_messages = conversation.messages[-3:] if len(conversation.messages) > 3 else conversation.messages
        cache_key = f"{user_query}_{str(last_messages)}"
        
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logger.info("Réponse trouvée dans le cache.")
            return cached_response
            
        language = detect_language(user_query)
        logger.info(f"Langue détectée pour la requête: {language}")
        
        logger.info(f"Recherche de contexte pour: {user_query[:50]}...")
        legal_context = pdf_indexer.get_relevant_context(user_query)
        if legal_context:
            logger.info(f"Contexte juridique trouvé (premiers 100 caractères): {legal_context[:100]}...")
        else:
            logger.info("Aucun contexte juridique trouvé.")
        
        messages_with_context = conversation.messages.copy()
        user_message_found = False
        for i in range(len(messages_with_context) - 1, -1, -1):
            if messages_with_context[i]["role"] == "user":
                user_message_found = True
                original_content = messages_with_context[i]['content']
                if legal_context:
                    if language == "arabic":
                        enhanced_message = f"""سؤال المستخدم: {original_content}\n\nالسياق القانوني التونسي الذي يجب مراعاته:\n{legal_context}\n\nأجب على السؤال بناءً على هذا السياق القانوني التونسي...""" 
                    else:
                        enhanced_message = f"""Question de l'utilisateur: {original_content}\n\nContexte juridique tunisien à prendre en compte:\n{legal_context}\n\nRéponds à la question en te basant sur ce contexte juridique tunisien...""" 
                    messages_with_context[i]["content"] = enhanced_message
                    logger.info(f"Message utilisateur enrichi avec contexte juridique en {language}.")
                else:
                    if language == "arabic":
                        enhanced_message = f"""سؤال المستخدم: {original_content}\n\nلم يتم العثور على معلومات محددة في قاعدة البيانات القانونية..."""
                    else:
                        enhanced_message = f"""Question de l'utilisateur: {original_content}\n\nAucune information spécifique n'a été trouvée dans la base de données juridique..."""
                    messages_with_context[i]["content"] = enhanced_message
                    logger.info(f"Message utilisateur enrichi avec instruction de réponse en {language} (aucun contexte trouvé).")
                break
        if not user_message_found:
            logger.warning("Aucun message utilisateur trouvé pour enrichissement.")

        logger.info(f"Envoi de la requête à Groq avec le modèle {GROQ_MODEL}. Messages: {len(messages_with_context)}")
        start_time = time.time()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages_with_context,
            temperature=0.3, max_tokens=1024, top_p=1, stream=False, stop=None
        )
        end_time = time.time()
        logger.info(f"Réponse reçue de Groq en {end_time - start_time:.2f} secondes.")
        response = completion.choices[0].message.content
        logger.info(f"Réponse générée par Groq (premiers 100 chars): {response[:100]}...")
        response_cache.set(cache_key, response)
        return response
    except HTTPException as http_exc:
        logger.error(f"HTTPException dans query_groq_api: {http_exc.status_code} - {http_exc.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.exception("Erreur inattendue dans query_groq_api.")
        raise HTTPException(status_code=500, detail=f"Erreur interne API Groq: {str(e)}")

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        logger.info(f"Création nouvelle conversation ID: {conversation_id}")
        conversations[conversation_id] = Conversation()
    else:
        conversation = conversations[conversation_id]
        if time.time() - conversation.last_activity > 3600: # 1 hour
            logger.info(f"Conversation {conversation_id} inactive, réinitialisation.")
            conversations[conversation_id] = Conversation() # Réinitialise
        else:
            logger.info(f"Conversation existante récupérée ID: {conversation_id}")
    return conversations[conversation_id]

@app.post("/chat/")
async def chat(input: UserInput, request: Request):
    logger.info(f"Requête /chat/ - ID: {input.conversation_id}, Msg: {input.message[:50]}...")
    if not input.message or not input.conversation_id:
        logger.error("Message ou conversation_id manquant dans /chat/")
        raise HTTPException(status_code=400, detail="Message et conversation_id obligatoires")
    try:
        conversation = get_or_create_conversation(input.conversation_id)
        if not conversation.active: # Devrait être géré par get_or_create_conversation
            raise HTTPException(status_code=400, detail="Session de chat inactive.")
        conversation.messages.append({"role": input.role, "content": input.message})
        conversation.update_last_activity()
        try:
            response = query_groq_api(conversation, input.message)
        except HTTPException as http_exc:
            logger.error(f"HTTPException de query_groq_api: {http_exc.status_code}", exc_info=True)
            if http_exc.status_code == 500 and "Groq API" in str(http_exc.detail):
                 raise HTTPException(status_code=503, detail="Service de génération de texte indisponible.")
            raise
        except Exception as e:
            logger.exception("Erreur non gérée query_groq_api depuis /chat/.")
            raise HTTPException(status_code=503, detail="Service temporairement indisponible.")
        conversation.messages.append({"role": "assistant", "content": response})
        logger.info(f"Réponse générée pour ID: {input.conversation_id}")
        return {"message": "Réponse générée", "response": response, "conversation_id": input.conversation_id, "language": detect_language(response)}
    except HTTPException as http_exc:
        logger.error(f"HTTPException dans /chat/: {http_exc.status_code}", exc_info=True)
        raise
    except Exception as e:
        logger.exception("Erreur majeure inattendue dans /chat/.")
        raise HTTPException(status_code=500, detail="Erreur interne majeure.")

@app.post("/reindex/")
async def reindex_documents_endpoint():
    logger.info("Requête reçue sur /reindex/")
    try:
        pdf_indexer.index_documents() # Appel à la réindexation complète
        logger.info("Réindexation des documents terminée avec succès via endpoint.")
        return {"message": "Documents réindexés avec succès!"}
    except Exception as e:
        logger.exception("Erreur lors de la réindexation via endpoint /reindex/")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la réindexation: {str(e)}")

@app.on_event("startup")
def startup_event():
    logger.info("Événement startup: Indexation des documents en cours...")
    try:
        pdf_indexer.index_documents() # Indexation complète au démarrage
        logger.info("Événement startup: Indexation des documents terminée.")
    except Exception as e:
        logger.exception("Erreur lors de l'indexation au démarrage.")

@app.get("/search/{query}")
def search(query: str):
    logger.info(f"Requête de recherche reçue pour: {query}")
    return pdf_indexer.search(query)

# ... (autres endpoints comme test-groq, clear_cache, feedback, generate_document, etc. peuvent rester ici)
@app.get("/test-groq/")
async def test_groq():
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return {"status": "success", "response": completion.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/clear_cache/")
async def clear_cache():
    response_cache.clear()
    logger.info("Cache de réponses vidé via endpoint.")
    return {"message": "Cache vidé avec succès"}

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackInput):
    feedback_dir = "feedback_data"
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, "feedback_log.csv")
    if not os.path.exists(feedback_file):
        with open(feedback_file, "w", encoding="utf-8") as f:
            f.write("timestamp,conversation_id,message_id,rating,comment\n")
    with open(feedback_file, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_comment = feedback.comment.replace(",", "\\,").replace("\n", "\\n")
        f.write(f"{timestamp},{feedback.conversation_id},{feedback.message_id},{feedback.rating},{safe_comment}\n")
    logger.info(f"Feedback reçu pour conversation {feedback.conversation_id}")
    return {"message": "Feedback enregistré"}

@app.get("/feedback/stats/")
async def get_feedback_stats():
    feedback_file = os.path.join("feedback_data", "feedback_log.csv")
    if not os.path.exists(feedback_file):
        return {"message": "Aucun feedback disponible", "stats": {}}
    ratings = []
    with open(feedback_file, "r", encoding="utf-8") as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try: ratings.append(int(parts[3]))
                except ValueError: continue
    if not ratings: return {"message": "Aucun feedback disponible", "stats": {}}
    avg_rating = sum(ratings) / len(ratings)
    rating_counts = {i: ratings.count(i) for i in range(1, 6)}
    return {"message": "Statistiques de feedback récupérées", "stats": {"total_feedbacks": len(ratings), "average_rating": avg_rating, "rating_distribution": rating_counts}}

@app.post("/generate_document/")
async def generate_document(request: DocumentRequest):
    supported_types = ["lettre_mise_en_demeure", "requete_simple", "procuration"]
    if request.document_type not in supported_types: raise HTTPException(400, "Type de document non supporté.")
    if request.language not in ["fr", "ar"]: raise HTTPException(400, "Langue non supportée.")
    template_dir = "document_templates"
    template_file = os.path.join(template_dir, f"{request.document_type}_{request.language}.txt")
    if not os.path.exists(template_file): raise HTTPException(404, "Template non trouvé")
    with open(template_file, "r", encoding="utf-8") as f: template = f.read()
    for key, value in request.parameters.items(): template = template.replace(f"{{{{{key}}}}}", str(value))
    filename = f"{request.document_type}_{request.language}_{time.strftime('%Y%m%d%H%M%S')}.txt"
    output_dir = "generated_documents"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f: f.write(template)
    return {"message": "Document généré", "document_content": template, "filename": filename}

@app.get("/document_templates/")
async def get_document_templates():
    template_dir = "document_templates"
    if not os.path.exists(template_dir): return {"templates": []}
    templates = []
    for file in os.listdir(template_dir):
        if file.endswith(".txt"): 
            parts = file.split("_")
            if len(parts) >= 2: templates.append({"type": parts[0], "language": parts[1].split(".")[0], "filename": file})
    return {"templates": templates}

# ------ Endpoint d'Upload de Document avec Indexation Incrémentale ------
@app.post("/upload_document/")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    language: str = Form("fr") 
):
    logger.info(f"Requête d'upload reçue pour conv ID: {conversation_id}, fichier: {file.filename}")
    try:
        if not file.filename:
            logger.error("Upload: Aucun nom de fichier fourni.")
            raise HTTPException(400, "Aucun nom de fichier fourni")

        filename = secure_filename(file.filename)
        if not filename:
            logger.error("Upload: Nom de fichier invalide après sécurisation.")
            raise HTTPException(400, "Nom de fichier invalide")

        allowed_extensions = {".pdf"} 
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.error(f"Upload: Type de fichier non supporté '{file_ext}'. Accepté: {allowed_extensions}")
            raise HTTPException(400, f"Type de fichier non supporté. Seuls les PDF sont acceptés pour l'indexation.")

        upload_dir = os.path.abspath(LEGAL_DOCS_FOLDER) 
        if not os.access(upload_dir, os.W_OK):
            logger.error(f"Upload: Permissions insuffisantes sur le dossier de destination {upload_dir}")
            raise HTTPException(500, f"Permissions insuffisantes sur le dossier de destination: {upload_dir}")

        file_location = os.path.join(upload_dir, filename)
        logger.info(f"Upload: Enregistrement du fichier vers: {file_location}")

        try:
            with open(file_location, "wb") as f:
                max_size = 50 * 1024 * 1024  # 50MB
                total_size = 0
                chunk_size = 1024 * 1024  # 1MB
                while content := await file.read(chunk_size):
                    total_size += len(content)
                    if total_size > max_size:
                        os.remove(file_location)
                        logger.error(f"Upload: Fichier trop volumineux (>{max_size}MB): {filename}")
                        raise HTTPException(413, f"Fichier trop volumineux. Maximum {max_size//(1024*1024)}MB")
                    f.write(content)
            logger.info(f"Upload: Fichier {filename} enregistré avec succès. Taille: {total_size} bytes.")
        except Exception as e:
            if os.path.exists(file_location):
                os.remove(file_location) 
            logger.exception(f"Upload: Erreur lors de l'écriture du fichier {filename}")
            raise HTTPException(500, "Erreur lors de l'enregistrement du fichier")

        summary = f"Fichier {filename} enregistré avec succès."
        if file_ext == ".pdf":
            logger.info(f"Le fichier {filename} est un PDF. Tentative d'ajout à l'index (incrémental)...")
            try:
                # Utilisation de la nouvelle méthode pour l'indexation incrémentale
                if pdf_indexer.add_single_document(file_location):
                    summary = f"Document PDF {filename} enregistré et ajouté à l'index avec succès."
                    logger.info(f"Document {filename} ajouté/mis à jour dans l'index (incrémental).")
                else:
                    summary = f"Document PDF {filename} enregistré, mais n'a pas pu être ajouté à l'index (contenu vide ou erreur)."
                    logger.warning(f"Échec de l'ajout incrémental de {filename} à l'index.")
            except Exception as e:
                logger.exception(f"Erreur lors de l'ajout incrémental du document {filename} à l'index.")
                summary = f"Document {filename} enregistré, mais une erreur est survenue lors de son ajout à l'index: {str(e)}"
        
        file_size_mb = os.path.getsize(file_location)/(1024*1024) if os.path.exists(file_location) else 0
        return {
            "status": "success",
            "filename": filename,
            "size": f"{file_size_mb:.2f}MB",
            "summary": summary,
            "conversation_id": conversation_id,
            "language": language
        }

    except HTTPException as http_exc:
        logger.error(f"Upload: HTTPException gérée: {http_exc.status_code} - {http_exc.detail}", exc_info=True)
        raise
    except Exception as e:
        logger.exception(f"Upload: Erreur inattendue et non gérée pour le fichier {file.filename if file else 'inconnu'}")
        raise HTTPException(500, f"Erreur interne du serveur lors de l'upload: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Démarrage du serveur Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

