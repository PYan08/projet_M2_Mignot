# Analyse de documents

Ce projet propose une interface pour charger des documents pour constituer une base de connaissance qui pourra être questionnée avec un grand modèle de langage (_LLM_).

# RAG UI Project

This project implements a Retrieval-Augmented Generation (RAG) system to analyze documents and provide insights using natural language processing.

## Table of Contents
- [Installation]
- [Usage]
- [Features]
- [Contributing]
- [License]

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PYan08/projet.git

2. **Navigate to the project directory:**
   ```bash
   cd projet

## Technologies Utilisées
- Streamlit
- Langchain
- LlamaIndex
- SQLite
- Pandas


## Usage

To run the application, use the following command:
bash
streamlit run app.py

## Features

- **Document Analysis:** Upload and analyze documents to extract insights.
- **Language Selection:** Choose the language for responses.
- **Framework Selection:** Select between different indexing frameworks.
- **Feedback Mechanism:** Provide feedback on the quality of responses.

## Fonctionnalités Implémentées

### Sélection de la Langue

L'application permet aux utilisateurs de choisir la langue de réponse parmi plusieurs options : Français, Anglais, Espagnol, Allemand. Cette fonctionnalité a été implémentée en utilisant un sélecteur de langue dans l'interface utilisateur, permettant à l'utilisateur de sélectionner la langue souhaitée pour les réponses générées par l'application.

python
language = st.selectbox(
    'Choisissez la langue de réponse:',
    ('Français', 'Anglais', 'Espagnol', 'Allemand')
)

framework_choice = st.radio(
    "Choisissez le framework d'indexation:",
    ('Langchain', 'LlamaIndex')
)

feedback = st.feedback("Like", "N/A", "Dislike")
if feedback:
    print(f"Feedback reçu: {feedback}")
    # Stocker le feedback dans la base de données SQLite
import sqlite3

def create_connection():
    conn = sqlite3.connect('feedback.db')
    return conn

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


## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

