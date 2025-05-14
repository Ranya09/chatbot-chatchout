import os
import pickle
import pdfplumber
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time

# Configurer le logger pour ce module
logger = logging.getLogger(__name__)

def table_to_markdown(table):
    """Convertit une liste de listes (tableau) en une chaîne Markdown."""
    markdown_table = ""
    if not table: return markdown_table
    header = table[0]
    markdown_table += "| " + " | ".join(str(h) if h is not None else '' for h in header) + " |\n"
    markdown_table += "| " + " | ".join("--- " * len(header)) + " |\n"
    for row in table[1:]:
        markdown_table += "| " + " | ".join(str(cell) if cell is not None else '' for cell in row) + " |\n"
    return markdown_table + "\n"

class PDFIndexer:
    def __init__(self, folder_path, cache_path="cache.pkl"):
        self.folder_path = folder_path
        self.cache_path = cache_path
        self.documents = [] # Liste de dictionnaires {"filename": str, "text": str, "modified_time": float}
        self.texts = []     # Liste des textes purs pour TF-IDF
        self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words=None, max_features=5000, ngram_range=(1, 2))
        self.tfidf_matrix = None
        logger.info(f"PDFIndexer initialisé pour le dossier: {folder_path} et cache: {cache_path}")

        if os.path.exists(cache_path):
            logger.info("Tentative de chargement de l'index depuis le cache...")
            self._load_cache()
            # La vérification des mises à jour est complexe avec l'ajout incrémental, 
            # pour l'instant, on se fie au cache ou à une réindexation manuelle.
            # self._check_for_updates() # Désactivé pour simplifier l'indexation incrémentale
        else:
            logger.info("Aucun cache trouvé. Indexation complète des documents requise au premier appel ou manuellement.")
            self.index_documents() # Indexe les documents existants au démarrage si pas de cache

    def _extract_text_and_tables(self, path):
        logger.debug(f"Début de l'extraction de texte et tableaux pour: {path}")
        full_content = ""
        try:
            with pdfplumber.open(path) as pdf:
                logger.info(f"Traitement de {len(pdf.pages)} pages pour {os.path.basename(path)}.")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_content += page_text + "\n\n"
                    tables = page.extract_tables()
                    if tables:
                        for table_data in tables:
                            if table_data:
                                markdown_table = table_to_markdown(table_data)
                                full_content += "\n--- DÉBUT TABLEAU ---\n"
                                full_content += markdown_table
                                full_content += "--- FIN TABLEAU ---\n\n"
        except Exception as e:
            logger.exception(f"Erreur lors de l'extraction de texte et tableaux du fichier PDF: {path}")
            return ""
        return full_content.strip()

    def _rebuild_tfidf(self):
        """Reconstruit le vectorizer et la matrice TF-IDF à partir de self.texts."""
        if self.texts:
            logger.info("Reconstruction de la matrice TF-IDF...")
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
                logger.info(f"Matrice TF-IDF reconstruite avec succès. Dimensions: {self.tfidf_matrix.shape}")
            except Exception as e:
                logger.exception("Erreur lors de la reconstruction de la matrice TF-IDF.")
                self.tfidf_matrix = None # Assurer un état cohérent
        else:
            logger.warning("Aucun texte à indexer. La matrice TF-IDF est vide.")
            self.tfidf_matrix = None
            # Réinitialiser le vectorizer s'il n'y a plus de textes pour éviter des états incohérents
            self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words=None, max_features=5000, ngram_range=(1, 2))

    def index_documents(self):
        logger.info(f"Début de l'indexation complète des documents dans {self.folder_path}")
        start_time_total = time.time()
        self.documents = []
        self.texts = []
        files_to_index = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
        logger.info(f"{len(files_to_index)} fichiers PDF trouvés pour l'indexation complète.")

        for filename in tqdm(files_to_index, desc="📄 Indexation PDF (Complète)"):
            path = os.path.join(self.folder_path, filename)
            content = self._extract_text_and_tables(path)
            if content:
                modified_time = os.path.getmtime(path)
                self.documents.append({
                    'filename': filename,
                    'text': content,
                    'modified_time': modified_time
                })
                self.texts.append(content)
            else:
                logger.warning(f"Aucun contenu extrait de {filename} lors de l'indexation complète.")
        
        self._rebuild_tfidf()
        self._save_cache()
        end_time_total = time.time()
        logger.info(f"Indexation complète terminée en {end_time_total - start_time_total:.2f} secondes. {len(self.documents)} documents indexés.")

    def add_single_document(self, file_path):
        """Ajoute ou met à jour un seul document PDF à l'index existant."""
        filename = os.path.basename(file_path)
        logger.info(f"Ajout/Mise à jour du document unique: {filename}")
        start_time = time.time()

        content = self._extract_text_and_tables(file_path)
        if not content:
            logger.warning(f"Aucun contenu extrait de {filename}. Le document ne sera pas ajouté/mis à jour.")
            return False

        modified_time = os.path.getmtime(file_path)
        
        # Vérifier si le document existe déjà pour le mettre à jour
        doc_exists = False
        for i, doc_info in enumerate(self.documents):
            if doc_info['filename'] == filename:
                logger.info(f"Le document {filename} existe déjà. Mise à jour du contenu et de modified_time.")
                self.documents[i]['text'] = content
                self.documents[i]['modified_time'] = modified_time
                self.texts[i] = content # Assumer que l'ordre est conservé
                doc_exists = True
                break
        
        if not doc_exists:
            logger.info(f"Nouveau document {filename}. Ajout à l'index.")
            self.documents.append({
                'filename': filename,
                'text': content,
                'modified_time': modified_time
            })
            self.texts.append(content)

        self._rebuild_tfidf() # Reconstruit TF-IDF avec le nouveau/mis à jour texte
        self._save_cache()
        end_time = time.time()
        logger.info(f"Document {filename} ajouté/mis à jour et index reconstruit en {end_time - start_time:.2f} secondes.")
        return True

    def remove_document(self, filename):
        """Supprime un document de l'index."""
        logger.info(f"Tentative de suppression du document: {filename} de l'index.")
        doc_found = False
        # Utiliser une compréhension de liste pour reconstruire documents et texts sans le fichier spécifié
        new_documents = []
        new_texts = []
        for doc_info in self.documents:
            if doc_info['filename'] == filename:
                doc_found = True
                logger.info(f"Document {filename} trouvé et marqué pour suppression.")
            else:
                new_documents.append(doc_info)
                new_texts.append(doc_info['text'])
        
        if doc_found:
            self.documents = new_documents
            self.texts = new_texts
            self._rebuild_tfidf()
            self._save_cache()
            logger.info(f"Document {filename} supprimé de l'index et index reconstruit.")
            return True
        else:
            logger.warning(f"Document {filename} non trouvé dans l'index. Aucune action de suppression.")
            return False

    def _save_cache(self):
        logger.info(f"Sauvegarde de l'index dans le cache: {self.cache_path}")
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'texts': self.texts, # Sauvegarder texts pour reconstruire si vectorizer change
                    'vectorizer_params': self.vectorizer.get_params(), # Sauvegarder les paramètres pour recréer
                    'vectorizer_vocabulary': getattr(self.vectorizer, 'vocabulary_', None),
                    'tfidf_matrix': self.tfidf_matrix
                }, f)
            logger.info("Cache sauvegardé avec succès.")
        except Exception as e:
            logger.exception(f"Erreur lors de la sauvegarde du cache dans {self.cache_path}")

    def _load_cache(self):
        logger.info(f"Chargement de l'index depuis le cache: {self.cache_path}")
        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.texts = data.get('texts', [])
                
                vectorizer_params = data.get('vectorizer_params')
                vocabulary = data.get('vectorizer_vocabulary')
                self.tfidf_matrix = data.get('tfidf_matrix')

                if vectorizer_params:
                    self.vectorizer = TfidfVectorizer(**vectorizer_params)
                    if self.texts and vocabulary is not None:
                        # Si le vocabulaire est sauvegardé et qu'il y a des textes, on peut essayer de reconstruire
                        # la matrice TF-IDF sans refaire fit_transform si les textes n'ont pas changé.
                        # Cependant, pour la robustesse, il est plus simple de refaire fit_transform.
                        self.vectorizer.vocabulary_ = vocabulary # Restaurer le vocabulaire
                        if self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] == len(self.texts):
                             logger.info("Vectorizer et TF-IDF matrix chargés depuis le cache.")
                        else:
                            logger.info("Vocabulaire restauré, mais la matrice TF-IDF sera reconstruite.")
                            self._rebuild_tfidf()
                    elif self.texts:
                         logger.info("Textes chargés, reconstruction de TF-IDF car vocabulaire non trouvé dans cache.")
                         self._rebuild_tfidf()
                    else:
                        logger.info("Vectorizer chargé, mais pas de textes ou de vocabulaire pour construire TF-IDF.")
                        self.tfidf_matrix = None
                else:
                    logger.warning("Paramètres du Vectorizer non trouvés dans le cache. Utilisation des valeurs par défaut.")
                    self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words=None, max_features=5000, ngram_range=(1, 2))
                    self._rebuild_tfidf() # Reconstruire si les paramètres par défaut sont utilisés

                if not self.documents or not self.texts:
                    logger.warning("Cache chargé mais documents ou textes vides. Une réindexation pourrait être nécessaire.")
                else:
                    logger.info(f"Cache chargé: {len(self.documents)} documents.")

        except FileNotFoundError:
            logger.error(f"Fichier cache {self.cache_path} non trouvé. L'indexation sera effectuée si des documents sont présents.")
        except Exception as e:
            logger.exception(f"Erreur lors du chargement du cache depuis {self.cache_path}. Réinitialisation de l'index.")
            self.documents = []
            self.texts = []
            self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words=None, max_features=5000, ngram_range=(1, 2))
            self.tfidf_matrix = None
            # Ne pas appeler index_documents() ici, laisser l'init ou un appel explicite le faire.

    def _check_for_updates(self): # Cette méthode devient moins pertinente avec l'ajout incrémental géré par l'application
        logger.info("Vérification des mises à jour des fichiers PDF (fonctionnalité _check_for_updates)...")
        # Cette logique devrait être revue si on veut un scan automatique en plus des ajouts/suppressions manuels.
        # Pour l'instant, on se fie aux appels explicites à add_single_document ou index_documents.
        pass

    def search(self, query, top_k=5):
        logger.debug(f"Recherche demandée pour la requête: '{query[:50]}...', top_k={top_k}")
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            logger.warning("Matrice TF-IDF non initialisée ou vide. Recherche impossible. Documents: %s", len(self.documents))
            # Optionnellement, tenter une réindexation si aucun document n'est chargé
            if not self.documents and os.path.exists(self.folder_path) and os.listdir(self.folder_path):
                logger.info("Tentative de réindexation car aucun document chargé et dossier non vide.")
                self.index_documents()
                if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
                    return {"error": "TF-IDF non initialisée ou aucun document indexable trouvé après tentative de réindexation."}
            else:
                return {"error": "TF-IDF non initialisée ou aucun document indexable trouvé."}
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            num_docs_available = self.tfidf_matrix.shape[0]
            actual_top_k = min(top_k, num_docs_available)
            # S'assurer que les indices sont valides pour self.documents
            if actual_top_k > len(self.documents):
                logger.error(f"Incohérence: actual_top_k ({actual_top_k}) > len(self.documents) ({len(self.documents)})")
                actual_top_k = len(self.documents)
            
            ranked_indices = similarities.argsort()[::-1][:actual_top_k]
            
            results = []
            for i in ranked_indices:
                if i < len(self.documents) and similarities[i] > 0.01:
                    results.append({
                        'filename': self.documents[i]['filename'],
                        'score': float(similarities[i])
                    })
                elif i >= len(self.documents):
                    logger.warning(f"Indice {i} hors limites pour self.documents lors de la recherche.")
            
            return results
        except Exception as e:
            logger.exception(f"Erreur lors de la recherche pour la requête: {query}")
            return {"error": f"Erreur lors de la recherche: {str(e)}"}

    def get_relevant_context(self, query, top_k=3):
        logger.debug(f"Obtention du contexte pertinent pour la requête: '{query[:50]}...', top_k={top_k}")
        results = self.search(query, top_k=top_k)
        
        if isinstance(results, dict) and "error" in results:
            logger.error(f"Erreur lors de la recherche de documents pour le contexte: {results['error']}")
            return ""
        if not results:
            logger.info(f"Aucun document pertinent trouvé pour la requête: '{query[:50]}...' lors de la recherche de contexte.")
            return ""

        context = ""
        chars_per_doc_limit = 1500 
        total_context_char_limit = 4000

        for result in results:
            if len(context) >= total_context_char_limit: break
            try:
                doc_content = next((doc['text'] for doc in self.documents if doc['filename'] == result['filename']), None)
                if doc_content:
                    context_to_add = f"\n--- Source: {result['filename']} (Score: {result['score']:.4f}) ---\n"
                    remaining_chars_for_doc = chars_per_doc_limit
                    remaining_total_chars = total_context_char_limit - len(context) - len(context_to_add)
                    chars_to_take_from_doc = min(len(doc_content), remaining_chars_for_doc, remaining_total_chars)
                    if chars_to_take_from_doc > 0:
                        context_to_add += doc_content[:chars_to_take_from_doc]
                        context += context_to_add
            except Exception as e:
                logger.exception(f"Erreur lors de la construction du contexte pour {result['filename']}.")
        return context.strip()

