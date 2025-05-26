import streamlit as st
from src.retriever import OphthalmoRetriever
from src.generator import OphthalmoGenerator
from src.ingest import DocumentIngestor
import os
import traceback

# Configuration de la page
st.set_page_config(
    page_title="RAG Ophtalmologie",
    page_icon="👁️",
    layout="wide"
)

st.title("🔍 Assistant RAG en Ophtalmologie")

# Vérification de la clé API
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Clé API GROQ manquante !")
    st.info("Ajoutez votre clé dans le fichier .env ")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.success("✅ API GROQ configurée")

# Bouton de réindexation
if st.sidebar.button("🔄 Indexer les documents"):
    with st.spinner("Indexation en cours..."):
        try:
            ingestor = DocumentIngestor()
            ingestor.ingest_all()
            st.sidebar.success("Documents indexés !")
        except Exception as e:
            st.sidebar.error(f"Erreur d'indexation: {str(e)}")

# Statistiques de la base
try:
    retriever = OphthalmoRetriever()
    stats = retriever.get_collection_stats()
    
    if 'error' not in stats:
        st.sidebar.markdown("### 📊 Statistiques")
        st.sidebar.metric("Documents totaux", stats.get('total_documents', 0))
        
        if stats.get('file_types'):
            st.sidebar.markdown("**Types de fichiers:**")
            for file_type, count in stats['file_types'].items():
                st.sidebar.text(f"• {file_type}: {count}")
                
except Exception as e:
    st.sidebar.error(f"Erreur de connexion à la base: {str(e)}")

# Interface principale
st.header("💬 Posez votre question")

query = st.text_area(
    "Votre question sur l'ophtalmologie:",
    placeholder="Ex: Quels sont les symptômes de la DMLA ?",
    height=100
)

# Options avancées
with st.expander("🔧 Options avancées"):
    n_results = st.slider("Nombre de documents à récupérer", 5, 20, 10)
    show_debug = st.checkbox("Afficher les informations de debug", False)

if st.button("🔍 Rechercher", type="primary"):
    if query.strip():
        with st.spinner("Recherche en cours..."):
            try:
                # Initialiser les composants
                if 'retriever' not in locals():
                    retriever = OphthalmoRetriever()
                generator = OphthalmoGenerator()
                
                # Récupération avec gestion d'erreur
                retrieved_docs = retriever.retrieve(query, n_results=n_results)
                
                if show_debug:
                    st.markdown("### 🔍 Debug - Documents récupérés")
                    for i, doc in enumerate(retrieved_docs[:5]):
                        st.markdown(f"**Doc {i+1}:** {doc['metadata'].get('source', 'Unknown')}")
                        
                        # Afficher tous les scores disponibles
                        scores_info = []
                        if 'combined_score' in doc:
                            scores_info.append(f"Score combiné: {doc['combined_score']}")
                        if 'semantic_score' in doc:
                            scores_info.append(f"Sémantique: {doc['semantic_score']}")
                        if 'keyword_score' in doc:
                            scores_info.append(f"Mots-clés: {doc['keyword_score']}")
                        
                        st.text(" | ".join(scores_info))
                        st.text(doc['content'][:200] + "...")
                        st.divider()
                
                if retrieved_docs:
                    # Génération
                    result = generator.generate_with_sources(query, retrieved_docs)
                    
                    # Affichage de la réponse
                    st.header("📝 Réponse")
                    st.write(result['response'])
                    
                    # Sources avec informations détaillées
                    st.header("📚 Sources utilisées")
                    
                    for i, source in enumerate(result['sources']):
                        with st.expander(f"📄 Source {i+1}: {source['source']} (Score: {source['score']})"):
                            
                            # Informations sur la source
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Score global", source['score'])
                            
                            with col2:
                                if 'semantic_score' in source:
                                    st.metric("Score sémantique", source['semantic_score'])
                            
                            with col3:
                                if 'keyword_score' in source:
                                    st.metric("Score mots-clés", source['keyword_score'])
                            
                            # Métadonnées supplémentaires
                            metadata_info = []
                            if 'page' in source:
                                metadata_info.append(f"Page: {source['page']}")
                            if 'type' in source:
                                metadata_info.append(f"Type: {source['type']}")
                            
                            if metadata_info:
                                st.text(" | ".join(metadata_info))
                            
                            # Contenu
                            st.markdown("**Extrait:**")
                            st.code(source['preview'], language="text")

                else:
                    st.warning("❌ Aucun document pertinent trouvé.")
                    st.info("💡 Essayez de reformuler votre question ou vérifiez que les documents sont bien indexés.")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de la recherche: {str(e)}")
                
                # Afficher la stack trace complète en mode debug
                if show_debug:
                    st.code(traceback.format_exc())
                
                # Suggestions de résolution
                st.markdown("### 🔧 Suggestions de résolution:")
                st.markdown("""
                1. **Vérifiez la base de données**: Cliquez sur '🔄 Indexer les documents'
                2. **Vérifiez les fichiers**: Assurez-vous que les dossiers `./data/pdfs` et `./data/docs` existent
                3. **Vérifiez la configuration**: GROQ_API_KEY dans le fichier .env
                4. **Réinitialisez**: Relancez l'application
                """)
    else:
        st.warning("⚠️ Veuillez saisir une question.")

# Section d'aide
with st.expander("ℹ️ Aide et conseils"):
    st.markdown("""
    ### Comment bien utiliser l'assistant:
    
    **Questions efficaces:**
    - "Quels sont les symptômes de la DMLA ?"
    - "Comment diagnostiquer un glaucome ?"
    - "Quels traitements pour la cataracte ?"
    
    **Structure des documents:**
    - Les documents doivent être dans `./data/pdfs/` (PDF) ou `./data/docs/` (DOCX/TXT)
    - Réindexez après ajout de nouveaux documents
    
    **Scores:**
    - **Score combiné**: Pertinence globale (0-1)
    - **Score sémantique**: Similarité du sens (0-1) 
    - **Score mots-clés**: Présence des termes recherchés (0-1)
    """)