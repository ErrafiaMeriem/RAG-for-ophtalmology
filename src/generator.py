import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Charge GROQ_API_KEY

class OphthalmoGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Clé GROQ_API_KEY manquante dans .env")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"

    def generate_response(self, query, retrieved_docs):
        # Construire le contexte avec gestion flexible des scores
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            # Gérer différents formats de score
            score = self.get_document_score(doc)
            source = doc['metadata'].get('source', 'Inconnu')
            
            context_parts.append(
                f"Document {i+1} ({source} - Score: {score}):\n{doc['content']}"
            )
        
        context = "\n\n".join(context_parts)

        prompt = f"""Tu es un assistant médical expert en ophtalmologie.
Tu dois répondre **principalement à partir des documents ci-dessous**.  
Ne donne pas de réponse inventée, mais tu peux **déduire ou regrouper des informations** si elles sont **partiellement présentes** dans les documents.

Si la réponse n’est pas du tout trouvable, indique-le clairement.

Ta réponse doit être :
- claire et structurée en français professionnel
- fidèle aux documents, mais synthétique si possible
### CONTEXTE :
{context}

### QUESTION :
{query}

### RÉPONSE :
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload)
            if response.status_code != 200:
                return f"❌ Erreur Groq : {response.status_code} - {response.text}"
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Exception lors de l'appel Groq : {str(e)}"

    def get_document_score(self, doc):
        """Récupère le score du document en gérant différents formats"""
        # Essayer différents champs de score
        if 'combined_score' in doc:
            return round(doc['combined_score'], 3)
        elif 'score' in doc:
            return round(doc['score'], 3)
        elif 'semantic_score' in doc:
            return round(doc['semantic_score'], 3)
        elif 'distance' in doc:
            # Convertir la distance en score (1 - distance)
            return round(max(0, 1 - doc['distance']), 3)
        else:
            return "N/A"

    def generate_with_sources(self, query, retrieved_docs):
        response = self.generate_response(query, retrieved_docs)

        sources = []
        for i, doc in enumerate(retrieved_docs):
            score = self.get_document_score(doc)
            source_info = {
                'source': doc['metadata'].get('source', 'Unknown'),
                'score': score,
                'preview': doc['content'][:200] + "..."
            }
            
            # Ajouter des informations supplémentaires si disponibles
            if 'semantic_score' in doc:
                source_info['semantic_score'] = round(doc['semantic_score'], 3)
            if 'keyword_score' in doc:
                source_info['keyword_score'] = round(doc['keyword_score'], 3)
            if doc['metadata'].get('page'):
                source_info['page'] = doc['metadata']['page']
            if doc['metadata'].get('file_type'):
                source_info['type'] = doc['metadata']['file_type']
                
            sources.append(source_info)

        return {
            'response': response,
            'sources': sources
        }