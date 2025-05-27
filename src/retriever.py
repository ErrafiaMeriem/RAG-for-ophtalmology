import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import re
from collections import defaultdict
import numpy as np
from .config import MultimodalConfig
from transformers import CLIPProcessor, CLIPModel
import torch

class OphthalmoRetriever:
    def __init__(self, vector_db_path="./vectordb",model_name: str = "openai/clip-vit-base-patch32"):
        try:
            # Initialiser ChromaDB
            self.client = chromadb.PersistentClient(path=vector_db_path)
            
            try:
                self.collection = self.client.get_collection(name="multimodal_content")
                print("✅ Collection multimodale récupérée")
            except:
                # Créer la collection si elle n'existe pas
                self.collection = self.client.create_collection(
                    name="multimodal_content",
                    metadata={"hnsw:space": "cosine", "model": MultimodalConfig.CLIP_MODEL_NAME}
                )
                print("✅ Collection multimodale créée")
            
            # Charger le modèle CLIP pour embeddings multimodaux
            self.clip_model = CLIPModel.from_pretrained(MultimodalConfig.CLIP_MODEL_NAME)
            self.clip_processor = CLIPProcessor.from_pretrained(MultimodalConfig.CLIP_MODEL_NAME)
            
            print(f"📦 Modèle CLIP chargé: {MultimodalConfig.CLIP_MODEL_NAME}")
            print(f"📏 Dimensions: {MultimodalConfig.EMBEDDING_DIMENSIONS}")
            
            # Mettre en mode évaluation
            self.clip_model.eval()
            
        except Exception as e:
            raise Exception(f"Erreur initialisation retriever multimodal: {e}")
    
    def preprocess_query(self, query):
        """Préprocesse la requête pour améliorer la recherche"""
        # Normaliser la casse et supprimer les caractères spéciaux
        processed_query = query.lower().strip()
        
        # Dictionnaire des synonymes médicaux courants
        synonyms = {
            'dmla': 'dégénérescence maculaire liée à l\'âge',
            'cataracte': 'opacification du cristallin',
            'glaucome': 'hypertension oculaire',
            'rétinopathie': 'maladie rétinienne',
            'myopie': 'trouble de la vision',
            'presbytie': 'trouble accommodation',
        }
        
        # Remplacer les acronymes par leurs définitions complètes
        for acronym, full_term in synonyms.items():
            if acronym in processed_query:
                processed_query = processed_query.replace(acronym, f"{acronym} {full_term}")
        
        return processed_query
    
    def extract_exact_phrases(self, query):
        """Extrait les expressions exactes à rechercher (entre guillemets ou expressions composées)"""
        exact_phrases = []
        
        # Chercher les expressions entre guillemets
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        exact_phrases.extend(quoted_phrases)
        
        # Identifier les expressions médicales composées importantes
        # Pattern pour capturer "mot + de/d' + mot" et autres expressions médicales
        medical_patterns = [
            r'\b\w+\s+d\'?\w+\b',  # myopie d'indice, cataracte de, etc.
            r'\b\w+\s+\w+(?:\s+\w+)?\b'  # expressions de 2-3 mots
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, query.lower())
            exact_phrases.extend(matches)
        
        # Filtrer pour garder seulement les expressions pertinentes
        stop_words = {'de', 'le', 'la', 'les', 'du', 'des', 'et', 'ou', 'à', 'dans', 'sur', 'avec', 'pour', 'par', 'ce', 'qui', 'que', 'est', 'sont'}
        
        relevant_phrases = []
        for phrase in exact_phrases:
            words = phrase.strip().split()
            if len(words) >= 2:
                # Garder l'expression si elle contient au moins un mot non-stop
                if not all(word.lower() in stop_words for word in words):
                    relevant_phrases.append(phrase.strip())
        
        # Supprimer les doublons et trier par longueur (plus long = plus spécifique)
        unique_phrases = list(set(relevant_phrases))
        unique_phrases.sort(key=len, reverse=True)
        
        return unique_phrases
    
    def retrieve_hybrid(self, query, n_results=15):
        """Méthode hybride combinant recherche sémantique et recherche exacte"""
        exact_phrases = self.extract_exact_phrases(query)
        
        if exact_phrases:
            print(f"🎯 Mode hybride activé pour expressions: {exact_phrases}")
            
            # 1. Recherche exacte pour chaque expression
            exact_results = []
            for phrase in exact_phrases:
                phrase_matches = self.search_exact_phrase(phrase, n_results=10)
                exact_results.extend(phrase_matches)
            
            # 2. Recherche sémantique normale
            semantic_results = self.retrieve(query, n_results=n_results, exact_match_boost=False)
            
            # 3. Fusionner et réorganiser les résultats
            combined = self.merge_exact_and_semantic_results(exact_results, semantic_results, exact_phrases)
            
            return combined[:n_results]
        else:
            # Recherche sémantique standard
            return self.retrieve(query, n_results=n_results)
    
    def merge_exact_and_semantic_results(self, exact_results, semantic_results, exact_phrases):
        """Fusionne les résultats de recherche exacte et sémantique"""
        merged = []
        seen_contents = set()
        
        # D'abord, ajouter tous les résultats avec expressions exactes (PRIORITÉ MAXIMALE)
        for exact_doc in exact_results:
            content_hash = hash(exact_doc['content'][:100])  # Hash pour déduplication
            if content_hash not in seen_contents:
                # Convertir le format de exact_results vers le format standard
                merged_doc = {
                    'content': exact_doc['content'],
                    'metadata': exact_doc['metadata'],
                    'semantic_score': 0.5,  # Score moyen
                    'keyword_score': exact_doc.get('exact_score', 0.9),  # Score élevé pour exact
                    'quality_score': exact_doc.get('quality_score', 0.5),
                    'exact_bonus': 0.3,  # Bonus élevé
                    'combined_score': 0.9 + exact_doc.get('exact_score', 0.0),  # Score très élevé
                    'distance': 0.1,  # Distance faible (très similaire)
                    'source_type': 'exact',
                    'has_exact_phrase': True
                }
                merged.append(merged_doc)
                seen_contents.add(content_hash)
        
        # Ensuite, ajouter les résultats sémantiques qui ne sont pas déjà présents
        for semantic_doc in semantic_results:
            content_hash = hash(semantic_doc['content'][:100])
            if content_hash not in seen_contents:
                merged.append(semantic_doc)
                seen_contents.add(content_hash)
        
        # Trier par score combiné (les résultats exacts seront en tête)
        merged.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return merged
    
    def search_exact_phrase(self, exact_phrase, n_results=10):
        """Recherche spécialisée pour une expression exacte - VERSION CORRIGÉE"""
        print(f"🔍 Recherche exacte pour: '{exact_phrase}'")
        
        # Récupérer TOUS les documents de la collection
        try:
            all_docs = self.collection.get(
                include=['documents', 'metadatas']
            )
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des documents: {e}")
            return []
        
        if not all_docs or not all_docs.get('documents'):
            print("❌ Aucun document trouvé dans la collection")
            return []
        
        exact_matches = []
        phrase_lower = exact_phrase.lower().strip()
        
        print(f"🔎 Recherche de '{phrase_lower}' dans {len(all_docs['documents'])} documents...")
        
        for i, doc in enumerate(all_docs['documents']):
            doc_lower = doc.lower()
            
            # Recherche exacte de l'expression
            if phrase_lower in doc_lower:
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                
                # Compter les occurrences
                occurrences = doc_lower.count(phrase_lower)
                
                # Calculer un score basé sur les occurrences et le contexte
                context_score = self.calculate_context_score(exact_phrase, doc)
                quality_score = self.calculate_quality_score(doc, metadata)
                
                # Score plus élevé pour les expressions exactes
                exact_score = min(1.0, 0.6 + (occurrences * 0.3) + context_score + quality_score)
                
                # Extraire un contexte autour de l'expression pour debug
                context_preview = self.extract_context_around_phrase(phrase_lower, doc_lower, 100)
                
                exact_matches.append({
                    'content': doc,
                    'metadata': metadata,
                    'exact_phrase': exact_phrase,
                    'occurrences': occurrences,
                    'exact_score': round(exact_score, 3),
                    'context_score': round(context_score, 3),
                    'quality_score': round(quality_score, 3),
                    'context_preview': context_preview
                })
                
                print(f"✅ Trouvé dans {metadata.get('source', 'Unknown')} - {occurrences} occurrence(s)")
                print(f"   Contexte: {context_preview}")
        
        # Trier par score et retourner les meilleurs
        exact_matches.sort(key=lambda x: x['exact_score'], reverse=True)
        
        print(f"🎯 Total: {len(exact_matches)} documents contenant '{exact_phrase}'")
        
        return exact_matches[:n_results]
    
    def extract_context_around_phrase(self, phrase, text, context_length=100):
        """Extrait le contexte autour d'une expression pour debug"""
        pos = text.find(phrase)
        if pos == -1:
            return ""
        
        start = max(0, pos - context_length)
        end = min(len(text), pos + len(phrase) + context_length)
        
        context = text[start:end]
        # Marquer l'expression trouvée
        context = context.replace(phrase, f"**{phrase}**")
        
        return context
    
    def normalize_semantic_score(self, distance, distance_function="cosine"):
        """Convertit la distance en score de similarité normalisé"""
        if distance_function == "cosine":
            # Distance cosinus: 0 = identique, 2 = opposé
            similarity = max(0, 1 - (distance / 2))
        elif distance_function == "euclidean":
            # Distance euclidienne: approximation avec fonction exponentielle
            similarity = np.exp(-distance / 2)
        elif distance_function == "dot_product":
            # Produit scalaire: plus élevé = plus similaire
            similarity = max(0, min(1, distance))
        else:
            # Par défaut, assume cosinus
            similarity = max(0, 1 - (distance / 2))
        
        return similarity
    
    def retrieve(self, query, n_results=15, exact_match_boost=True):
        """Récupère les documents les plus pertinents avec scoring amélioré"""
        
        # Préprocesser la requête
        processed_query = self.preprocess_query(query)
        
        # Identifier les expressions exactes
        exact_phrases = self.extract_exact_phrases(query)
        if exact_phrases:
            print(f"🎯 Expressions exactes recherchées: {exact_phrases}")
        
        # Extraire l'embedding de la requête
        with torch.no_grad():
            inputs = self.clip_processor(text=processed_query, return_tensors="pt", padding=True, truncation=True)
            query_embedding = self.clip_model.get_text_features(**inputs).squeeze().numpy()


        
        # Recherche sémantique standard
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 3,  # Récupérer plus pour filtrer
            include=['documents', 'metadatas', 'distances']
        )
        
        # Recherche par expressions exactes si nécessaire
        exact_results = []
        if exact_match_boost and exact_phrases:
            for phrase in exact_phrases:
                phrase_results = self.search_exact_phrase(phrase, n_results=n_results)
                exact_results.extend(phrase_results)
        
        # Combiner les résultats
        all_candidates = []
        seen_contents = set()
        
        # Ajouter résultats exacts EN PREMIER (priorité maximale)
        for exact_match in exact_results:
            content_hash = hash(exact_match['content'][:100])
            if content_hash not in seen_contents:
                all_candidates.append((
                    exact_match['content'], 
                    exact_match['metadata'], 
                    0.1,  # Distance très faible pour exact match
                    'exact'
                ))
                seen_contents.add(content_hash)
        
        # Ajouter résultats sémantiques
        if semantic_results['documents'][0]:
            for doc, metadata, distance in zip(
                semantic_results['documents'][0],
                semantic_results['metadatas'][0], 
                semantic_results['distances'][0]
            ):
                content_hash = hash(doc[:100])
                if content_hash not in seen_contents:
                    all_candidates.append((doc, metadata, distance, 'semantic'))
                    seen_contents.add(content_hash)
        
        if not all_candidates:
            return []
        
        # Calculer les scores pour tous les candidats
        distances = [candidate[2] for candidate in all_candidates]
        min_distance = min(distances) if distances else 0
        max_distance = max(distances) if distances else 1
        distance_range = max_distance - min_distance if max_distance > min_distance else 1
        
        print(f"📈 Distance range: {min_distance:.3f} - {max_distance:.3f} (range: {distance_range:.3f})")
        
        retrieved_docs = []
        
        for doc, metadata, distance, source_type in all_candidates:
            # Calculer différents scores
            semantic_score = self.normalize_semantic_score(distance, "cosine")
            
            # Score sémantique relatif
            relative_semantic_score = 1 - ((distance - min_distance) / distance_range)
            final_semantic_score = (semantic_score * 0.7) + (relative_semantic_score * 0.3)
            
            # Score de pertinence textuelle (CRITIQUE pour expressions exactes)
            keyword_score = self.calculate_keyword_score(query.lower(), doc.lower())
            
            # Score de qualité du chunk
            quality_score = self.calculate_quality_score(doc, metadata)
            
            # Bonus si trouvé par recherche exacte
            exact_bonus = 0.4 if source_type == 'exact' else 0
            
            # Score combiné avec BOOST MAJEUR pour correspondances exactes
            has_exact_phrase = exact_phrases and any(phrase.lower() in doc.lower() for phrase in exact_phrases)
            
            if has_exact_phrase:
                # BOOST MAJEUR si contient expression exacte
                combined_score = (
                    final_semantic_score * 0.2 +   # Réduire sémantique
                    keyword_score * 0.7 +           # BOOST mots-clés exacts
                    quality_score * 0.1 +
                    exact_bonus                     # Bonus recherche exacte
                )
            else:
                # Score normal
                combined_score = (
                    final_semantic_score * 0.5 +
                    keyword_score * 0.4 +
                    quality_score * 0.1 +
                    exact_bonus
                )
            
            retrieved_docs.append({
                'content': doc,
                'metadata': metadata,
                'semantic_score': round(final_semantic_score, 3),
                'keyword_score': round(keyword_score, 3),
                'quality_score': round(quality_score, 3),
                'exact_bonus': round(exact_bonus, 3),
                'combined_score': round(combined_score, 3),
                'distance': round(distance, 3),
                'source_type': source_type,
                'has_exact_phrase': has_exact_phrase
            })
        
        # Trier par score combiné
        retrieved_docs.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Dédoublonner mais préserver les documents avec expressions exactes
        filtered_docs = self.deduplicate_results_with_exact_priority(retrieved_docs, exact_phrases)
        
        # Retourner les top résultats
        final_results = filtered_docs[:n_results]
        
        # Log détaillé
        print(f"\n🔍 Requête: '{query}'")
        print(f"📊 Candidats: {len(all_candidates)} | Après filtrage: {len(final_results)}")
        
        exact_found = sum(1 for doc in final_results if doc['has_exact_phrase'])
        if exact_phrases:
            print(f"🎯 Documents avec expressions exactes: {exact_found}/{len(final_results)}")
        
        for i, doc in enumerate(final_results[:5]):
            source = doc['metadata'].get('source', 'Unknown')
            scores = f"🧠{doc['semantic_score']} 🔑{doc['keyword_score']} 🎯{doc['combined_score']}"
            exact_marker = "⭐" if doc['has_exact_phrase'] else ""
            preview = doc['content'][:100].replace('\n', ' ')
            
            print(f"  {i+1}. {exact_marker} {source} ({scores})")
            print(f"     📝 {preview}...")
        
        return final_results
    
    def calculate_keyword_score(self, query, content):
        """Calcule un score basé sur la présence de mots-clés et expressions exactes"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # 1. BONUS MAJEUR pour expressions exactes
        exact_phrases = self.extract_exact_phrases(query)
        exact_phrase_score = 0
        
        for phrase in exact_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in content_lower:
                # Gros bonus pour expression exacte trouvée
                exact_phrase_score += 0.8
                print(f"✅ Expression exacte trouvée: '{phrase}' dans le contenu")
            else:
                print(f"❌ Expression exacte manquante: '{phrase}'")
        
        # Normaliser le score des expressions exactes
        if exact_phrases:
            exact_phrase_score = exact_phrase_score / len(exact_phrases)
        
        # 2. Score traditionnel des mots-clés individuels
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))  # Mots de 3+ caractères
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        if not query_words:
            return exact_phrase_score
        
        # Compter les mots communs
        matches = len(query_words.intersection(content_words))
        keyword_score = matches / len(query_words)
        
        # 3. Bonus pour fréquence des mots-clés
        frequency_bonus = 0
        for word in query_words:
            if word in content_lower:
                count = content_lower.count(word)
                frequency_bonus += min(0.1 * np.log(count + 1), 0.2)
        
        # 4. Bonus pour proximité des mots-clés
        proximity_bonus = self.calculate_proximity_bonus(query_words, content_lower)
        
        # 5. Score final combiné avec priorité aux expressions exactes
        individual_word_score = keyword_score + frequency_bonus + proximity_bonus
        
        # Si on a des expressions exactes, elles dominent le score
        if exact_phrases:
            total_score = (exact_phrase_score * 0.8) + (individual_word_score * 0.2)
        else:
            total_score = individual_word_score
        
        return min(total_score, 1.0)
    
    def calculate_proximity_bonus(self, query_words, content_lower):
        """Calcule un bonus basé sur la proximité des mots-clés dans le texte"""
        words_in_content = re.findall(r'\b\w+\b', content_lower)
        query_positions = {}
        
        # Trouver les positions de chaque mot-clé
        for word in query_words:
            positions = [i for i, w in enumerate(words_in_content) if w == word]
            if positions:
                query_positions[word] = positions
        
        if len(query_positions) < 2:
            return 0
        
        # Calculer la distance moyenne entre les mots-clés
        proximity_bonus = 0
        word_pairs = [(w1, w2) for w1 in query_positions for w2 in query_positions if w1 != w2]
        
        for w1, w2 in word_pairs:
            for pos1 in query_positions[w1]:
                for pos2 in query_positions[w2]:
                    distance = abs(pos1 - pos2)
                    if distance <= 10:  # Mots proches (dans une fenêtre de 10 mots)
                        proximity_bonus += 0.1 / (distance + 1)
        
        return min(proximity_bonus, 0.3)
    
    def calculate_quality_score(self, content, metadata):
        """Calcule un score de qualité du chunk amélioré"""
        score = 0.3  # Score de base
        
        # Bonus pour la longueur appropriée
        length = len(content)
        if 300 <= length <= 800:
            score += 0.3  # Zone optimale
        elif 200 <= length < 300 or 800 < length <= 1200:
            score += 0.2  # Zone acceptable
        elif length > 1200:
            score += 0.1  # Texte long, moins accessible
        
        # Bonus si c'est un PDF (souvent plus structuré)
        if metadata.get('file_type') == 'pdf':
            score += 0.1
        
        # Bonus pour certaines sources
        source = metadata.get('source', '').lower()
        if any(keyword in source for keyword in ['guide', 'manuel', 'cours', 'référentiel']):
            score += 0.15
        
        # Analyse de la structure du contenu
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) >= 2:
            score += 0.1
        
        # Vérifier la présence de listes ou structure
        if re.search(r'[•\-\*]\s|^\d+\.|\n\s*\d+\.', content, re.MULTILINE):
            score += 0.05
        
        # Malus si le chunk semble être du bruit
        noise_ratio = len(re.findall(r'[^\w\s\-.,;:!?()]', content)) / max(len(content), 1)
        if noise_ratio > 0.3:
            score -= 0.3
        elif noise_ratio > 0.2:
            score -= 0.1
        
        # Malus pour texte trop répétitif
        words = content.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.3:
                score -= 0.2
        
        return max(0, min(score, 1))
    
    def calculate_context_score(self, phrase, content):
        """Calcule un score basé sur le contexte autour de l'expression exacte"""
        phrase_lower = phrase.lower()
        content_lower = content.lower()
        
        if phrase_lower not in content_lower:
            return 0
        
        context_score = 0
        
        # Trouver toutes les positions de l'expression
        start = 0
        while True:
            pos = content_lower.find(phrase_lower, start)
            if pos == -1:
                break
            
            # Analyser le contexte autour (50 caractères avant et après)
            context_start = max(0, pos - 50)
            context_end = min(len(content), pos + len(phrase_lower) + 50)
            context = content_lower[context_start:context_end]
            
            # Bonus si l'expression est dans une phrase complète
            if '.' in context or pos == 0 or content_lower[pos-1] in ' \n\t':
                context_score += 0.1
            
            # Bonus si entourée de mots-clés médicaux
            medical_keywords = ['traitement', 'symptôme', 'diagnostic', 'thérapie', 'patient', 'maladie']
            for keyword in medical_keywords:
                if keyword in context:
                    context_score += 0.05
            
            start = pos + 1
        
        return min(context_score, 0.4)
    
    def text_similarity(self, text1, text2):
        """Calcule la similarité entre deux textes avec Jaccard amélioré"""
        # Utiliser des n-grammes pour une meilleure détection de similarité
        def get_ngrams(text, n=3):
            words = re.findall(r'\b\w+\b', text.lower())
            if len(words) < n:
                return set([' '.join(words)])
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
        
        # Combiner mots uniques et trigrammes
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        ngrams1 = get_ngrams(text1, 3)
        ngrams2 = get_ngrams(text2, 3)
        
        # Similarité des mots
        word_jaccard = (len(words1.intersection(words2)) / 
                       len(words1.union(words2))) if words1.union(words2) else 0
        
        # Similarité des n-grammes
        ngram_jaccard = (len(ngrams1.intersection(ngrams2)) / 
                        len(ngrams1.union(ngrams2))) if ngrams1.union(ngrams2) else 0
        
        # Moyenne pondérée
        return (word_jaccard * 0.4) + (ngram_jaccard * 0.6)
    
    def deduplicate_results_with_exact_priority(self, docs, exact_phrases, similarity_threshold=0.7):
        """Supprime les doublons en préservant les documents avec expressions exactes"""
        if len(docs) <= 1:
            return docs
        
        filtered = []
        
        for doc in docs:
            is_duplicate = False
            
            for existing in filtered:
                content_similarity = self.text_similarity(doc['content'], existing['content'])
                same_source = (doc['metadata'].get('source') == 
                             existing['metadata'].get('source'))
                
                threshold = similarity_threshold
                if same_source:
                    threshold = 0.6
                
                if content_similarity > threshold:
                    # Logique de priorité pour les doublons
                    doc_has_exact = doc['has_exact_phrase']
                    existing_has_exact = existing['has_exact_phrase']
                    
                    if doc_has_exact and not existing_has_exact:
                        # Le nouveau doc a une phrase exacte, l'ancien non -> remplacer
                        filtered.remove(existing)
                        filtered.append(doc)
                    elif not doc_has_exact and existing_has_exact:
                        # L'ancien a une phrase exacte, le nouveau non -> garder l'ancien
                        pass
                    elif doc['combined_score'] > existing['combined_score']:
                        # Même statut d'expression exacte -> garder le meilleur score
                        filtered.remove(existing)
                        filtered.append(doc)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(doc)
        
        return filtered
    
    def get_collection_stats(self):
        """Retourne des statistiques sur la collection"""
        try:
            count = self.collection.count()
            
            # Échantillon pour analyser les métadonnées
            sample = self.collection.get(limit=100)
            
            sources = defaultdict(int)
            file_types = defaultdict(int)
            
            for metadata in sample['metadatas']:
                if metadata:
                    sources[metadata.get('source', 'Unknown')] += 1
                    file_types[metadata.get('file_type', 'Unknown')] += 1
            
            return {
                'total_documents': count,
                'sources': dict(sources),
                'file_types': dict(file_types)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def search_by_source(self, query, source_filter=None, n_results=10):
        """Recherche dans des sources spécifiques"""
        if source_filter:
            # Récupérer tous les documents de la source
            try:
                all_docs = self.collection.get(
                    where={"source": source_filter},
                    include=['documents', 'metadatas']
                )
            except:
                # Si le filtrage where ne fonctionne pas, faire le filtrage manuellement
                all_results = self.retrieve(query, n_results=50)
                filtered = [doc for doc in all_results 
                           if doc['metadata'].get('source') == source_filter]
                return filtered[:n_results]
            
            if not all_docs['documents']:
                return []
            
            # Effectuer la recherche seulement dans ces documents
            all_results = self.retrieve(query, n_results=50)
            
            # Filtrer les résultats par source
            filtered = [doc for doc in all_results 
                       if doc['metadata'].get('source') == source_filter]
            
            return filtered[:n_results]
        else:
            return self.retrieve(query, n_results)
    
    def debug_embedding_similarity(self, query, top_n=3):
        """Fonction de debug pour analyser les similarités d'embeddings"""
        processed_query = self.preprocess_query(query)
        with torch.no_grad():
            inputs = self.clip_processor(text=processed_query, return_tensors="pt", padding=True, truncation=True)
            query_embedding = self.clip_model.get_text_features(**inputs).squeeze().numpy()


        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"🔬 Debug embeddings pour: '{query}'")
        print(f"📝 Requête préprocessée: '{processed_query}'")
        print(f"🧮 Dimension embedding: {len(query_embedding)}")
        
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0], 
            results['metadatas'][0]
        )):
            print(f"\n--- Résultat {i+1} ---")
            print(f"Distance brute: {distance:.6f}")
            print(f"Score normalisé: {self.normalize_semantic_score(distance):.3f}")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Contenu: {doc[:200]}...")
        
        return results
    
    def debug_exact_search(self, phrase):
        """Fonction de debug spécifique pour la recherche d'expressions exactes"""
        print(f"\n🔍 DEBUG RECHERCHE EXACTE pour: '{phrase}'")
        print("=" * 60)
        
        # 1. Vérifier l'extraction d'expressions
        extracted = self.extract_exact_phrases(phrase)
        print(f"🎯 Expressions extraites: {extracted}")
        
        # 2. Recherche manuelle dans tous les documents
        try:
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            print(f"📚 Total documents dans la collection: {len(all_docs['documents'])}")
        except Exception as e:
            print(f"❌ Erreur accès collection: {e}")
            return
        
        phrase_lower = phrase.lower().strip()
        matches_found = 0
        
        for i, doc in enumerate(all_docs['documents']):
            doc_lower = doc.lower()
            if phrase_lower in doc_lower:
                matches_found += 1
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                source = metadata.get('source', f'Doc_{i}')
                
                # Compter occurrences
                count = doc_lower.count(phrase_lower)
                
                # Extraire contexte
                pos = doc_lower.find(phrase_lower)
                start = max(0, pos - 100)
                end = min(len(doc), pos + len(phrase_lower) + 100)
                context = doc[start:end].replace('\n', ' ')
                
                print(f"\n✅ MATCH #{matches_found}")
                print(f"   📄 Source: {source}")
                print(f"   🔢 Occurrences: {count}")
                print(f"   📍 Position: {pos}")
                print(f"   📝 Contexte: ...{context}...")
                
                if matches_found >= 5:  # Limiter l'affichage
                    break
        
        print(f"\n📊 RÉSUMÉ: {matches_found} documents contiennent '{phrase}'")
        
        # 3. Tester la méthode search_exact_phrase
        print(f"\n🔧 Test de search_exact_phrase:")
        exact_results = self.search_exact_phrase(phrase, n_results=5)
        print(f"   Résultats retournés: {len(exact_results)}")
        
        # 4. Tester la méthode retrieve_hybrid
        print(f"\n🔧 Test de retrieve_hybrid:")
        hybrid_results = self.retrieve_hybrid(phrase, n_results=5)
        print(f"   Résultats retournés: {len(hybrid_results)}")
        
        exact_in_hybrid = sum(1 for r in hybrid_results if r.get('has_exact_phrase', False))
        print(f"   Avec expression exacte: {exact_in_hybrid}")
        
        return {
            'phrase': phrase,
            'extracted_phrases': extracted,
            'manual_matches': matches_found,
            'exact_search_results': len(exact_results),
            'hybrid_results': len(hybrid_results),
            'exact_in_hybrid': exact_in_hybrid
        }
    
    def verify_phrase_in_collection(self, phrase):
        """Vérification simple si une phrase existe dans la collection"""
        print(f"🔎 Vérification de la présence de: '{phrase}'")
        
        try:
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            phrase_lower = phrase.lower().strip()
            
            found_docs = []
            for i, doc in enumerate(all_docs['documents']):
                if phrase_lower in doc.lower():
                    metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                    found_docs.append({
                        'index': i,
                        'source': metadata.get('source', 'Unknown'),
                        'occurrences': doc.lower().count(phrase_lower)
                    })
            
            if found_docs:
                print(f"✅ Phrase trouvée dans {len(found_docs)} documents:")
                for doc_info in found_docs[:10]:  # Limiter l'affichage
                    print(f"   - {doc_info['source']}: {doc_info['occurrences']} fois")
            else:
                print(f"❌ Phrase '{phrase}' non trouvée dans la collection")
            
            return found_docs
            
        except Exception as e:
            print(f"❌ Erreur lors de la vérification: {e}")
            return []

# Fonctions utilitaires pour debugging
def test_myopie_indice(retriever):
    """Test spécifique pour 'myopie d'indice'"""
    print("\n" + "="*80)
    print("🧪 TEST SPÉCIFIQUE: MYOPIE D'INDICE")
    print("="*80)
    
    # Test 1: Vérification directe
    print("\n1️⃣ VÉRIFICATION DIRECTE:")
    found_docs = retriever.verify_phrase_in_collection("myopie d'indice")
    
    # Test 2: Debug complet
    print("\n2️⃣ DEBUG COMPLET:")
    debug_results = retriever.debug_exact_search("myopie d'indice")
    
    # Test 3: Recherche avec différentes variantes
    print("\n3️⃣ TEST VARIANTES:")
    variants = [
        "myopie d'indice",
        "myopie d indice", 
        "myopie indice",
        '"myopie d\'indice"'
    ]
    
    for variant in variants:
        print(f"\n--- Test: '{variant}' ---")
        results = retriever.retrieve_hybrid(variant, n_results=3)
        exact_found = sum(1 for r in results if r.get('has_exact_phrase', False))
        print(f"Résultats: {len(results)}, Avec expression exacte: {exact_found}")
    
    return debug_results

# Exemple d'utilisation pour diagnostiquer le problème
def diagnostic_complet(retriever, phrase="myopie d'indice"):
    """Diagnostic complet pour identifier pourquoi une phrase n'est pas trouvée"""
    print(f"\n🏥 DIAGNOSTIC COMPLET pour: '{phrase}'")
    print("="*80)
    
    # Étape 1: Vérifier si la phrase existe
    print("\n📋 ÉTAPE 1: Vérification de l'existence")
    found_docs = retriever.verify_phrase_in_collection(phrase)
    
    if not found_docs:
        print("❌ La phrase n'existe pas dans la collection - Problème de données")
        return
    
    # Étape 2: Tester l'extraction d'expressions
    print("\n📋 ÉTAPE 2: Test d'extraction d'expressions")
    extracted = retriever.extract_exact_phrases(phrase)
    print(f"Expressions extraites: {extracted}")
    
    if phrase.lower() not in [e.lower() for e in extracted]:
        print("❌ L'expression n'est pas correctement extraite - Problème de regex")
    
    # Étape 3: Tester la recherche exacte
    print("\n📋 ÉTAPE 3: Test de recherche exacte")
    exact_results = retriever.search_exact_phrase(phrase, n_results=5)
    print(f"Résultats de recherche exacte: {len(exact_results)}")
    
    # Étape 4: Tester la recherche hybride
    print("\n📋 ÉTAPE 4: Test de recherche hybride")
    hybrid_results = retriever.retrieve_hybrid(phrase, n_results=5)
    exact_in_hybrid = sum(1 for r in hybrid_results if r.get('has_exact_phrase', False))
    print(f"Résultats hybrides: {len(hybrid_results)}")
    print(f"Avec expression exacte: {exact_in_hybrid}")
    
    # Étape 5: Analyse des scores
    print("\n📋 ÉTAPE 5: Analyse des scores")
    if hybrid_results:
        for i, result in enumerate(hybrid_results[:3]):
            print(f"Résultat {i+1}:")
            print(f"  - Score combiné: {result['combined_score']}")
            print(f"  - Score keyword: {result['keyword_score']}")
            print(f"  - A expression exacte: {result['has_exact_phrase']}")
            print(f"  - Source: {result['metadata'].get('source', 'Unknown')}")
    
    return {
        'phrase_exists': len(found_docs) > 0,
        'correctly_extracted': phrase.lower() in [e.lower() for e in extracted],
        'exact_search_works': len(exact_results) > 0,
        'hybrid_search_works': exact_in_hybrid > 0,
        'found_in_sources': [doc['source'] for doc in found_docs]
    }