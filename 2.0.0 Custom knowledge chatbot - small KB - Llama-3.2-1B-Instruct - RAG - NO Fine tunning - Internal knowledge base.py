import os
import spacy
import pandas as pd
import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import keyboard  # To detect escape key press

# --------------------------------------------------------------- MODEL DEFINITION --------------------------------------------------------------------------------------
# --- Configure Environment for Windows Symlink Compatibility ---
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warnings for Windows

# --- Hugging Face Authentication ---
login("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")  # Replace with your Hugging Face token

# --- Setup NLP and LLaMA Model ---
nlp = spacy.load("en_core_web_sm")  # Load spaCy for text preprocessing
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token="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")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token="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").to(device)

# Set up the text generation pipeline with LLaMA model
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, device=0 if device == torch.device("cuda") else -1)

# --- Knowledge Base Setup with Documents ---
documents = [
    """
    Pidriš is a small, hilly-mountainous settlement, located just a few kilometers from Mount Raduša, which represents the natural border between “Bosnia” and “Herzegovina”. Locals often like to joke that Pidriš represents the “i” in the name of the country Bosnia and Herzegovina.
    """,
    """
    Thanks to the temperate-continental climate and the altitude of 950 meters, you can enjoy snow for a significant part of the year. Extremely clean air, an abundance of natural spring water, flora, fauna and extremely fertile soil, in addition to the aforementioned snow, are the characteristics of this settlement. Of the natural spring water springs, the Lučica spring is certainly worth mentioning, next to which, under the canopy, there was once a pleasant area for a nature trip.If you are lucky, during a walk in nature you may encounter a deer, a rabbit, or if you are not lucky, you may encounter a bear who likes to walk in early autumn… but he will not do anything to you, he is a vegetarian… I guess. We have not given him a name yet.
    """,
    """
    An interesting fact is that this small town had its own beer factory. Due to its quality, which is largely due to the natural spring water, the beer was recognizable both in the nearby municipalities, namely G.V-Uskoplje and Rama, and throughout the Central Bosnian County. The beer was called, how different from "Pidriško pivo", and the production itself has since been discontinued.
    """,
    """
    This village is also adorned with 25 box-shaped tombstones that date back to the early Middle Ages, and are located near the church of St. Anthony of Padua, who is also the patron saint of the town. During the celebration of the patron saint of the town, the village has many times more visitors than Pidriš itself has inhabitants. The branch church of St. Anthony of Padua was built in 1972, at the initiative of the villagers of Pidriš and the neighboring village of Mačkovac.
    """,
    """
    An overview of the population census throughout history is given below: 1885 - 106 inhabitants,1895 - 136 inhabitants,1910. year - 164 inhabitants,1961. year - 313 inhabitants,1971. year - 314 inhabitants,1981. year - 290 inhabitants,1991. year - 302 inhabitants,2013. year - 307 inhabitants
    """,
    """
    The natural and geographical position makes this settlement a suitable place for engaging in various sports and recreational activities, with an emphasis on the winter period of the year, which is supported by the construction and opening of the "Raduša" Ski Center, which, along with numerous trails, has two restaurants and apartments. In addition to engaging in winter sports and recreational activities such as skiing, snowboarding, sledding, the aforementioned natural and geographical features (clean air, rich spring waters, forests, wildlife ...) are an ideal combination for hiking along well-trodden hiking trails.
    """,
    """
    The route to the mountain lodge "Zekina Gruda" (1338 m) is worth highlighting, from which you can enjoy wonderful views of the inhabited area. In addition to the hiking route on „Zekina Gruda“, two more hiking routes are recommended, i.e. hiking to the highest peak of the mountain Raduša "Idovac" (1956 m) from which there is a simultaneous view of G.V.-Uskoplje, Bugojno, Kupres and Tomislavgrad, which is truly worth the climb. The last but not least recommended hiking route is a visit to „Draševo“, from the viewpoint of which there is a really beautiful view of „Ramsko Jezero“, which will surely leave a lasting impression on you.
    """,
    """
    If you are fond of football, the village has two football fields, one of which is concrete and the other is grass.
    """,
    """
    Chess is a game that is often identified with Pidriš, thanks to the outstanding results of local chess players in numerous tournaments, in which representatives from the largest cities in the country (such as Mostar, which has a population of about 113,000) participate and are defeated by a team from Pidriš.
    """,
    """
    On weekends, when the vast majority of the population is free from everyday obligations, the hunting group "Vepar" becomes active, and in addition to hunting, it participates in feeding game during the winter and in various other commendable activities. At organized hunter gatherings, there is always a good mood, and it is not uncommon for people to sing.
    """
]

# --- Setup Sentence Transformer and FAISS Index ---
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Load SentenceTransformer for embeddings
document_embeddings = model_encoder.encode(documents)  # Generate embeddings for each document

# Initialize FAISS index for fast document retrieval
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# --------------------------------------------------------------- FUNCTIONS DEFINITIONS --------------------------------------------------------------------------------------

# Step 1: Preprocess text for uniformity (for both query and documents)
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Step 2: Define a function to retrieve the most relevant document based on the user query
def retrieve_document(query):
    processed_query = preprocess_text(query)
    query_embedding = model_encoder.encode([processed_query])

    # Search in FAISS index for the closest document
    k = 1  # Retrieve top 1 most relevant document
    distances, indices = index.search(query_embedding, k)

    # Retrieve the document index and text
    retrieved_doc_index = indices[0][0]
    return retrieved_doc_index, documents[retrieved_doc_index]  # Return document index and text

# Step 3: Define function to generate response based on retrieved document
def generate_response(query, use_conversation_history=True):
    # Retrieve the relevant document for context
    retrieved_doc_index, retrieved_doc_text = retrieve_document(query)
    print(f"Retrieved Document Index: {retrieved_doc_index}")
    print(f"Document Preview: {retrieved_doc_text[:100]}...")  # Print the first 100 characters of the document for preview

    # Build context for model
    contextual_input = f"User query: {query}\nDocument context: {retrieved_doc_text}\nAnswer:"

    # Generate response using LLaMA model
    response = generator(
        contextual_input,
        max_new_tokens=512,  # Adjust as necessary
        num_return_sequences=1,
        truncation=True,
        do_sample=True,
        temperature=0.7
    )

    # Extract the answer from the generated text
    generated_text = response[0]["generated_text"]
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()
    else:
        generated_text = generated_text.strip()

    return generated_text

# --------------------------------------------------------------- MAIN LOOP --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Chatbot ready! Type your questions. Press 'Esc' to exit.")

    while True:
        if keyboard.is_pressed("esc"):
            print("\nExiting chatbot. Goodbye!")
            break

        print("-----------------------------------------------------------------")
        user_query = input("You: ")
        print("User Query:", user_query)

        # Generate and print the response
        chatbot_response = generate_response(user_query)
        print("[FINAL CHATBOT ANSWER]:", chatbot_response)
