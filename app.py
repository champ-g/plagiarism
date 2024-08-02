import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read files
def read_file(file):
    return file.read().decode('utf-8')

# Function to vectorize text
def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

# Function to calculate similarity
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

# Streamlit UI
st.title("Plagiarism Checker")

uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)

if uploaded_files:
    student_files = [file.name for file in uploaded_files]
    student_notes = [read_file(file) for file in uploaded_files]

    vectors = vectorize(student_notes)
    s_vectors = list(zip(student_files, vectors))
    plagiarism_results = set()

    def check_plagiarism():
        global s_vectors
        for student_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((student_a, text_vector_a))
            del new_vectors[current_index]
            for student_b, text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                student_pair = sorted((student_a, student_b))
                score = (student_pair[0], student_pair[1], sim_score)
                plagiarism_results.add(score)
        return plagiarism_results

    results = check_plagiarism()
    for data in results:
        st.write(f"Similarity data: {data}")
