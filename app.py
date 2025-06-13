import os
import tempfile
import sqlite3
import streamlit as st
import pandas as pd

# Connexion à la base de données SQLite
def create_connection():
    conn = sqlite3.connect('feedback.db')
    return conn

# Création de la table feedbacks si elle n'existe pas
def create_feedback_table(conn):
    query = """
    CREATE TABLE IF NOT EXISTS feedbacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rating TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    conn.execute(query)
    conn.commit()

# Ajouter un feedback à la base de données
def add_feedback(conn, rating, question, answer):
    query = """
    INSERT INTO feedbacks (rating, question, answer)
    VALUES (?, ?, ?)
    """
    conn.execute(query, (rating, question, answer))
    conn.commit()

# Ajouter un bouton radio pour choisir le framework
framework_choice = st.radio(
    "Choisissez le framework d'indexation:",
    ('Langchain', 'LlamaIndex')
)

# Importer les fonctions appropriées en fonction du framework choisi
if framework_choice == 'Langchain':
    from rag.langchain import answer_question, delete_file_from_store, store_pdf_file
else:
    from rag.llamaindex import answer_question, delete_file_from_store, store_pdf_file

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

def main():
    # Connexion à la base de données
    conn = create_connection()
    create_feedback_table(conn)

    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA en les chargeant dans l'application. Puis posez toutes vos questions.")

    # Ajouter un sélecteur de langue
    language = st.selectbox(
        'Choisissez la langue de réponse:',
        ('Français', 'Anglais', 'Espagnol', 'Allemand')
    )

    # Téléversement de fichiers multiples
    uploaded_files = st.file_uploader(
        label="Déposez vos fichiers ici ou chargez-les",
        type=None,
        accept_multiple_files=True
    )

    # S'il y a des fichiers, on affiche leurs noms et tailles
    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_in_kb:.2f}"
            })

            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, "temp.pdf")
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)

        df = pd.DataFrame(file_info)
        st.table(df)

    # Gestion de la suppression de documents
    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        delete_file_from_store(name)

    # Champ de question
    question = st.text_input("Votre question ici")

    # Bouton pour lancer l’analyse
    if st.button("Analyser"):
        model_response = answer_question(question, language)
        st.text_area("Zone de texte, réponse du modèle", value=model_response, height=200)

        # Ajouter un mécanisme de feedback
        feedback = st.feedback("Like", "N/A", "Dislike")
        if feedback:
            print(f"Feedback reçu: {feedback}")
            add_feedback(conn, feedback, question, model_response)
    else:
        st.text_area("Zone de texte, réponse du modèle", value="", height=200)

    # Fermer la connexion à la base de données
    conn.close()

if __name__ == "__main__":
    main()
