import os
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import shutil
import re
import hashlib
import base64
import io
import numpy as np
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
import pandas as pd
import cv2
import camelot
import torch
import clip


class DocumentIngestor:
    def __init__(self, vector_db_path="./vectordb", reset_db=False, collection_name="multimodal_content"):
        print("Initialisation de l'ingesteur CLIP multimodal...")
        
        
          # Stocker le nom de collection
        self.collection_name = collection_name 
        
        # Option pour réinitialiser la base
        if reset_db and os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
            print("Base vectorielle réinitialisée")
        
        # Initialiser ChromaDB
        self.client = chromadb.PersistentClient(path=vector_db_path)
        
        # Gérer la collection avec gestion d'erreur robuste
        self.setup_collection()
        
        # Charger CLIP une seule fois pour tout
        print("Chargement du modèle CLIP...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")
        
        try:
            # Charger CLIP - un seul modèle pour tout !
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✓ Modèle CLIP chargé avec succès")
            
            # Tester les dimensions d'embedding
            test_text = clip.tokenize(["test"]).to(self.device)
            with torch.no_grad():
                test_features = self.clip_model.encode_text(test_text)
                self.embedding_dim = test_features.shape[1]
            print(f"✓ Dimension des embeddings CLIP: {self.embedding_dim}")
            
        except ImportError as e:
            print(f"Erreur d'import CLIP: {e}")
            print("Installation requise: pip install torch torchvision clip-by-openai")
            raise
        
        # Splitter optimisé
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
            length_function=len,
        )
        
        # Dossiers de sauvegarde
        self.images_dir = os.path.join(os.path.dirname(vector_db_path), "extracted_images")
        self.tables_dir = os.path.join(os.path.dirname(vector_db_path), "extracted_tables")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
    
    def setup_collection(self):
        """Configure une seule collection pour tout le contenu multimodal"""
        try:
            existing_collections = self.client.list_collections()
            existing_names = [c.name for c in existing_collections]
            
            print(f"Collections existantes: {existing_names}")
            
            if self.collection_name in existing_names:
                print(f"Collection '{self.collection_name}' trouvée")
                self.collection = self.client.get_collection(self.collection_name)
            else:
                print(f"Création de la collection multimodale '{self.collection_name}'")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine", "description": "Contenu multimodal avec CLIP"}
                )
                 
        except Exception as e:
            print(f"Erreur avec la collection: {e}")
            # En cas d'erreur, tenter de supprimer et recréer
            try:
                print(f"Tentative de suppression de la collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
                print("Collection supprimée")
            except Exception as delete_error:
                print(f"Erreur lors de la suppression: {delete_error}")
            
            print(f"Création d'une nouvelle collection '{self.collection_name}'...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "Contenu multimodal avec CLIP"}
            )
        print("✓ Collection multimodale configurée")
    
    def list_collections(self):
        """Liste toutes les collections disponibles"""
        try:
            collections = self.client.list_collections()
            print("Collections disponibles:")
            for i, collection in enumerate(collections):
                count = collection.count()
                print(f"  {i+1}. {collection.name} ({count} éléments)")
            return collections
        except Exception as e:
            print(f"Erreur lors du listage des collections: {e}")
            return []
    
    def get_collection_info(self):
        """Affiche les informations de la collection actuelle"""
        try:
            count = self.collection.count()
            print(f"Collection '{self.collection_name}': {count} éléments")
            
            # Obtenir quelques exemples pour voir les types de contenu
            if count > 0:
                sample = self.collection.get(limit=5)
                if sample['metadatas']:
                    content_types = {}
                    for metadata in sample['metadatas']:
                        content_type = metadata.get('content_type', 'unknown')
                        content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    print("Types de contenu dans la collection:")
                    for content_type, count in content_types.items():
                        print(f"  - {content_type}: {count}+ éléments")
            
        except Exception as e:
            print(f"Erreur lors de la récupération des infos: {e}")
    def get_clip_embedding(self, content, content_type="text"):
        """Génère un embedding CLIP selon le type de contenu"""
        try:
            with torch.no_grad():
                if content_type == "image" and isinstance(content, Image.Image):
                    # Pour les images PIL directement
                    image_tensor = self.clip_preprocess(content).unsqueeze(0).to(self.device)
                    features = self.clip_model.encode_image(image_tensor)
                    return features[0].cpu().numpy().tolist()
                
                elif content_type == "image" and isinstance(content, str):
                    # Pour les chemins d'images
                    image = Image.open(content)
                    image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    features = self.clip_model.encode_image(image_tensor)
                    return features[0].cpu().numpy().tolist()
                
                else:
                    # Pour le texte (texte brut, tableaux, descriptions)
                    if isinstance(content, str):
                        # Truncate le texte si trop long (CLIP a une limite de tokens)
                        text_tokens = clip.tokenize([content[:500]], truncate=True).to(self.device)
                        features = self.clip_model.encode_text(text_tokens)
                        return features[0].cpu().numpy().tolist()
                    else:
                        # Fallback si le contenu n'est pas une string
                        text_tokens = clip.tokenize([str(content)[:500]], truncate=True).to(self.device)
                        features = self.clip_model.encode_text(text_tokens)
                        return features[0].cpu().numpy().tolist()
                        
        except Exception as e:
            print(f"Erreur génération embedding CLIP: {e}")
            # Retourner un embedding vide de la bonne dimension
            return [0.0] * self.embedding_dim
    
    def analyze_image_with_clip(self, pil_image: Image.Image) -> str:
        """Analyse une image avec CLIP pour générer une description"""
        try:
            # Descriptions possibles pour le contenu médical
            possible_descriptions = [
                "medical diagram showing anatomical structures",
                "x-ray or radiological medical image", 
                "microscopic tissue or cell image",
                "surgical procedure or medical intervention",
                "medical equipment or diagnostic device",
                "patient examination or clinical photo",
                "medical chart, graph or data visualization",
                "pharmaceutical product or medication",
                "medical textbook page or educational material",
                "laboratory test results or medical report"
            ]
            
            # Encoder l'image
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            text_tokens = clip.tokenize(possible_descriptions).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Calculer les similarités
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Prendre les 2 meilleures descriptions
                top_2_indices = similarities[0].topk(2).indices
                descriptions = []
                
                for idx in top_2_indices:
                    confidence = similarities[0][idx].item()
                    desc = possible_descriptions[idx.item()]
                    descriptions.append(f"{desc} (confidence: {confidence:.2f})")
                
                return " | ".join(descriptions)
                
        except Exception as e:
            print(f"  ⚠️ Erreur analyse CLIP: {e}")
            return "medical image or document"
    
    def clean_text(self, text: str) -> str:
        """Nettoie et améliore le texte extrait"""
        if not text:
            return ""
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Corriger les retours à la ligne mal formatés
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        
        # Normaliser les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Garder seulement les lignes substantielles
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        
        return '\n'.join(lines).strip()
    
    def process_docs(self, docs_folder: str):
        """Traite les fichiers .txt du dossier docs"""
        documents = []
        
        if not os.path.exists(docs_folder):
            print(f"Dossier {docs_folder} non trouvé")
            return documents
            
        txt_files = [f for f in os.listdir(docs_folder) if f.endswith('.txt')]
        print(f"Traitement de {len(txt_files)} fichiers texte...")
        
        for filename in txt_files:
            try:
                print(f"Traitement de {filename}...")
                file_path = os.path.join(docs_folder, filename)
                
                # Lire le contenu du fichier texte
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Nettoyer le texte
                cleaned_content = self.clean_text(content)
                
                if cleaned_content.strip():
                    # Créer un document similaire aux PDFs
                    doc = type("Doc", (), {
                        "page_content": cleaned_content,
                        "metadata": {
                            "source": filename,
                            "file_type": "txt",
                            "content_type": "text",
                            "file_size": len(content),
                            "char_count": len(cleaned_content)
                        }
                    })
                    documents.append(doc)
                    print(f"  ✓ Texte extrait: {len(cleaned_content)} caractères")
                else:
                    print(f"  ⚠️ Fichier vide ou sans contenu substantiel: {filename}")
                    
            except UnicodeDecodeError:
                # Essayer avec d'autres encodages
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                    cleaned_content = self.clean_text(content)
                    
                    if cleaned_content.strip():
                        doc = type("Doc", (), {
                            "page_content": cleaned_content,
                            "metadata": {
                                "source": filename,
                                "file_type": "txt",
                                "content_type": "text",
                                "encoding": "latin-1",
                                "file_size": len(content),
                                "char_count": len(cleaned_content)
                            }
                        })
                        documents.append(doc)
                        print(f"  ✓ Texte extrait (latin-1): {len(cleaned_content)} caractères")
                    
                except Exception as e:
                    print(f"  ⚠️ Erreur d'encodage pour {filename}: {e}")
                    
            except Exception as e:
                print(f"Erreur avec {filename}: {e}")
                
        return documents
    
    def extract_images_from_pdf(self, pdf_path: str, filename: str) -> List[Dict]:
        """Extrait et analyse les images d'un PDF"""
        images_data = []
        
        try:
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extraire l'image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        
                        # Convertir en PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY ou RGB
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                            pix1 = None
                        
                        pix = None
                        
                        # Filtrer les images trop petites
                        if pil_image.width < 100 or pil_image.height < 100:
                            continue
                        
                        # Sauvegarder l'image
                        image_filename = f"{filename}_page{page_num+1}_img{img_index+1}.png"
                        image_path = os.path.join(self.images_dir, image_filename)
                        pil_image.save(image_path)
                        
                        # OCR sur l'image
                        ocr_text = ""
                        try:
                            ocr_text = pytesseract.image_to_string(pil_image, lang='fra+eng')
                            ocr_text = self.clean_text(ocr_text)
                        except Exception as e:
                            print(f"  ⚠️ OCR échoué pour {image_filename}: {e}")
                        
                        # Analyser l'image avec CLIP
                        image_description = self.analyze_image_with_clip(pil_image)
                        
                        # Générer l'embedding CLIP directement de l'image
                        image_embedding = self.get_clip_embedding(pil_image, "image")
                        
                        images_data.append({
                            'filename': image_filename,
                            'path': image_path,
                            'page': page_num + 1,
                            'source_doc': filename,
                            'ocr_text': ocr_text,
                            'description': image_description,
                            'width': pil_image.width,
                            'height': pil_image.height,
                            'pil_image': pil_image,  # Garder l'image pour l'embedding
                            'embedding': image_embedding
                        })
                        
                        print(f"  📷 Image extraite: {image_filename} ({pil_image.width}x{pil_image.height})")
                        
                    except Exception as e:
                        print(f"  ⚠️ Erreur extraction image {img_index} page {page_num}: {e}")
            
            pdf_doc.close()
            
        except Exception as e:
            print(f"Erreur lors de l'extraction d'images de {filename}: {e}")
        
        return images_data
    
    def extract_tables_from_pdf(self, pdf_path: str, filename: str) -> List[Dict]:
        """Extrait les tableaux d'un PDF"""
        tables_data = []
        
        # Méthode Camelot
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(camelot_tables):
                if table.accuracy > 50:
                    table_filename = f"{filename}_table_{i+1}.csv"
                    table_path = os.path.join(self.tables_dir, table_filename)
                    
                    table.to_csv(table_path)
                    df = table.df
                    table_text = self.dataframe_to_text(df)
                    
                    # Générer l'embedding CLIP du texte du tableau
                    table_embedding = self.get_clip_embedding(table_text, "text")
                    
                    tables_data.append({
                        'filename': table_filename,
                        'path': table_path,
                        'source_doc': filename,
                        'method': 'camelot',
                        'accuracy': table.accuracy,
                        'text_content': table_text,
                        'rows': len(df),
                        'cols': len(df.columns),
                        'page': table.page,
                        'embedding': table_embedding
                    })
                    
                    print(f"  📊 Tableau extrait: {table_filename} ({len(df)} lignes)")
        
        except Exception as e:
            print(f"  ⚠️ Extraction tableaux échouée: {e}")
        
        return tables_data
    
    def dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convertit un DataFrame en représentation textuelle"""
        if df.empty:
            return ""
        
        df_clean = df.copy().fillna("")
        lines = []
        
        # En-têtes
        if len(df_clean) > 0:
            headers = [str(col).strip() for col in df_clean.iloc[0]]
            if any(headers):
                lines.append("COLONNES: " + " | ".join(headers))
                start_row = 1
            else:
                start_row = 0
        
        # Données (limiter à 15 lignes pour éviter les embeddings trop longs)
        for idx in range(start_row, min(len(df_clean), 15)):
            row_data = [str(cell).strip() for cell in df_clean.iloc[idx]]
            if any(row_data):
                lines.append("LIGNE: " + " | ".join(row_data))
        
        return "\n".join(lines)
    
    def create_stable_id(self, content: str, source: str, content_type: str, index: int) -> str:
        """Crée un ID stable"""
        content_hash = hashlib.sha256(f"{content}{source}{content_type}".encode()).hexdigest()[:12]
        return f"{content_type}_{source}_{index}_{content_hash}"
    
    def process_pdfs(self, pdf_folder: str):
        """Traite les PDFs"""
        documents = []
        all_images = []
        all_tables = []
        
        if not os.path.exists(pdf_folder):
            print(f"Dossier {pdf_folder} non trouvé")
            return documents, all_images, all_tables
            
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        print(f"Traitement de {len(pdf_files)} fichiers PDF...")
        
        for filename in pdf_files:
            try:
                print(f"Traitement de {filename}...")
                file_path = os.path.join(pdf_folder, filename)
                
                # Extraction du texte
                try:
                    pdf_doc = fitz.open(file_path)
                    docs = []
                    
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc.load_page(page_num)
                        text = page.get_text()
                        
                        if text.strip():
                            doc = type("Doc", (), {
                                "page_content": self.clean_text(text),
                                "metadata": {
                                    "source": filename,
                                    "page": page_num + 1,
                                    "file_type": "pdf",
                                    "content_type": "text"
                                }
                            })
                            docs.append(doc)
                    
                    pdf_doc.close()
                    print(f"  ✓ Texte extrait: {len(docs)} pages")
                    
                except Exception as e:
                    print(f"  ⚠️ Extraction texte échouée: {e}")
                    docs = []
                
                # Extraction des images
                images = self.extract_images_from_pdf(file_path, filename.replace('.pdf', ''))
                all_images.extend(images)
                
                # Extraction des tableaux
                tables = self.extract_tables_from_pdf(file_path, filename.replace('.pdf', ''))
                all_tables.extend(tables)
                
                documents.extend(docs)
                
            except Exception as e:
                print(f"Erreur avec {filename}: {e}")
                
        return documents, all_images, all_tables
    
    def ingest_all(self):
        """Lance l'ingestion complète avec CLIP"""
        print("=== DÉBUT DE L'INGESTION CLIP MULTIMODALE ===")
        
        # Traiter les PDFs
        print("\n📄 TRAITEMENT DES PDFs...")
        pdf_docs, images, tables = self.process_pdfs('./data/pdfs')
        
        # Traiter les fichiers .txt
        print("\n📝 TRAITEMENT DES FICHIERS TEXTE...")
        txt_docs = self.process_docs('./data/docs')
        
        # Combiner tous les documents texte
        all_text_docs = pdf_docs + txt_docs
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Documents texte: {len(pdf_docs)}")
        print(f"  Documents TXT: {len(txt_docs)}")
        print(f"  Total documents texte: {len(all_text_docs)}")
        print(f"  Images: {len(images)}")
        print(f"  Tableaux: {len(tables)}")
        
        if not pdf_docs and not images and not tables:
            print("ATTENTION: Aucun contenu trouvé!")
            return
        
        # Ingestion dans une seule collection
        print(f"\n🚀 INGESTION AVEC CLIP...")
        
        all_items = []
        
        # 1. Traiter le texte (PDF + TXT)
        if all_text_docs:
            chunks = self.text_splitter.split_documents(all_text_docs)
            print(f"Chunks texte: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.get_clip_embedding(chunk.page_content, "text")
                    
                    item = {
                        'id': self.create_stable_id(chunk.page_content, 
                                                  chunk.metadata.get('source', 'unknown'), 
                                                  'text', i),
                        'content': chunk.page_content,
                        'embedding': embedding,
                        'metadata': {**chunk.metadata, 'content_type': 'text'}
                    }
                    all_items.append(item)
                    
                except Exception as e:
                    print(f"Erreur chunk texte {i}: {e}")
        
        # 2. Traiter les images
        if images:
            print(f"Images à traiter: {len(images)}")
            
            for i, img_data in enumerate(images):
                try:
                    # Créer le contenu textuel combiné
                    content_parts = []
                    if img_data['description']:
                        content_parts.append(f"Description: {img_data['description']}")
                    if img_data['ocr_text']:
                        content_parts.append(f"Texte OCR: {img_data['ocr_text']}")
                    
                    content = "\n".join(content_parts) if content_parts else img_data['description']
                    
                    if not content.strip():
                        continue
                    
                    # Utiliser l'embedding déjà calculé de l'image
                    embedding = img_data['embedding']
                    
                    metadata = {
                        'content_type': 'image',
                        'source': img_data['source_doc'],
                        'page': img_data['page'],
                        'image_filename': img_data['filename'],
                        'image_path': img_data['path'],
                        'width': img_data['width'],
                        'height': img_data['height'],
                        'has_ocr_text': bool(img_data['ocr_text'].strip()),
                    }
                    
                    item = {
                        'id': self.create_stable_id(content, img_data['source_doc'], 'image', i),
                        'content': content,
                        'embedding': embedding,
                        'metadata': metadata
                    }
                    all_items.append(item)
                    
                except Exception as e:
                    print(f"Erreur image {i}: {e}")
        
        # 3. Traiter les tableaux
        if tables:
            print(f"Tableaux à traiter: {len(tables)}")
            
            for i, table_data in enumerate(tables):
                try:
                    content = table_data['text_content']
                    if not content.strip():
                        continue
                    
                    # Utiliser l'embedding déjà calculé du tableau
                    embedding = table_data['embedding']
                    
                    metadata = {
                        'content_type': 'table',
                        'source': table_data['source_doc'],
                        'table_filename': table_data['filename'],
                        'table_path': table_data['path'],
                        'extraction_method': table_data['method'],
                        'accuracy': table_data.get('accuracy', 0),
                        'rows': table_data.get('rows', 0),
                        'cols': table_data.get('cols', 0),
                        'page': table_data.get('page', 0)
                    }
                    
                    item = {
                        'id': self.create_stable_id(content, table_data['source_doc'], 'table', i),
                        'content': content,
                        'embedding': embedding,
                        'metadata': metadata
                    }
                    all_items.append(item)
                    
                except Exception as e:
                    print(f"Erreur tableau {i}: {e}")
        
        # 4. Ingestion dans ChromaDB
        print(f"\n💾 INGESTION DE {len(all_items)} ÉLÉMENTS...")
        
        success_count = 0
        batch_size = 50
        
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i+batch_size]
            
            try:
                ids = [item['id'] for item in batch]
                documents = [item['content'] for item in batch]
                embeddings = [item['embedding'] for item in batch]
                metadatas = [item['metadata'] for item in batch]
                
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                success_count += len(batch)
                print(f"  Progression: {success_count}/{len(all_items)}")
                
            except Exception as e:
                print(f"Erreur batch {i//batch_size}: {e}")
        
        print(f"✓ Ingestion terminée: {success_count}/{len(all_items)} éléments")
        
        # Statistiques finales
        total_count = self.collection.count()
        print(f"\n=== STATISTIQUES FINALES ===")
        print(f"Total dans la collection: {total_count} éléments")
    
    def search(self, query: str, top_k: int = 10, content_types: List[str] = None) -> Dict:
        """Recherche multimodale avec CLIP"""
        print(f"Recherche: '{query}'")
        
        # Générer l'embedding de la requête avec CLIP
        query_embedding = self.get_clip_embedding(query, "text")
        
        # Recherche dans la collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Filtrer par type de contenu si spécifié
            if content_types:
                filtered_results = {
                    'ids': [[]],
                    'documents': [[]],
                    'metadatas': [[]],
                    'distances': [[]]
                }
                
                for i, metadata in enumerate(results['metadatas'][0]):
                    if metadata.get('content_type') in content_types:
                        filtered_results['ids'][0].append(results['ids'][0][i])
                        filtered_results['documents'][0].append(results['documents'][0][i])
                        filtered_results['metadatas'][0].append(results['metadatas'][0][i])
                        if 'distances' in results:
                            filtered_results['distances'][0].append(results['distances'][0][i])
                
                return filtered_results
            
            return results
            
        except Exception as e:
            print(f"Erreur recherche: {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]]}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingesteur CLIP multimodal')
    parser.add_argument('--reset', action='store_true', help='Réinitialiser la base')
    parser.add_argument('--search', type=str, help='Recherche')
    parser.add_argument('--types', nargs='+', choices=['text', 'image', 'table'], 
                       help='Types de contenu à rechercher')
    parser.add_argument('--collection', type=str, default='multimodal_content',
                       help='Nom de la collection (défaut: multimodal_content)')
    parser.add_argument('--list', action='store_true', help='Lister les collections')
    parser.add_argument('--info', action='store_true', help='Infos sur la collection')
    
    args = parser.parse_args()
    
    # Initialiser l'ingesteur CLIP
    ingestor = DocumentIngestor(reset_db=args.reset,collection_name=args.collection)
    
    if args.list:
         # Lister les collections
        ingestor.list_collections()
    elif args.info:
        # Infos sur la collection
        ingestor.get_collection_info()
    elif args.search:
        # Mode recherche
        results = ingestor.search(args.search, content_types=args.types)
        
        if results['documents'] and results['documents'][0]:
            print(f"\n=== RÉSULTATS ({len(results['documents'][0])}) ===")
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                content_type = metadata.get('content_type', 'unknown')
                source = metadata.get('source', 'N/A')
                
                print(f"\n{i+1}. [{content_type.upper()}] Source: {source}")
                
                if content_type == 'image':
                    print(f"   Image: {metadata.get('image_filename', 'N/A')}")
                    print(f"   Dimensions: {metadata.get('width')}x{metadata.get('height')}")
                elif content_type == 'table':
                    print(f"   Tableau: {metadata.get('table_filename', 'N/A')}")
                    print(f"   Taille: {metadata.get('rows')}x{metadata.get('cols')}")
                elif content_type == 'text':
                    print(f"   Page: {metadata.get('page', 'N/A')}")
                
                print(f"   Contenu: {doc[:300]}...")
                
                if 'distances' in results and results['distances']:
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # Convertir distance en similarité
                    print(f"   Similarité: {similarity:.3f}")
        else:
            print("Aucun résultat trouvé")
    else:
        # Mode ingestion
        ingestor.ingest_all()