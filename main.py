import sqlite3
import streamlit as st
import pandas as pd
from langchain_community.llms.ollama import Ollama
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Initialize the cache
if 'question_cache' not in st.session_state:
    st.session_state.question_cache = {}

def csv_to_sqlite(csv_file, db_name, tablename):
    df = pd.read_csv(csv_file)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    def create_table_from_df(df, tablename):
        col_types = []
        for col, dtype in df.dtypes.items():
            if dtype == 'int64':
                col_type = 'INTEGER'
            elif dtype == 'float64':
                col_type = 'REAL'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = 'DATE'
            else:
                col_type = 'TEXT'
            col_types.append(f'"{col}" {col_type}')

        col_definitions = ", ".join(col_types)
        create_table_query = f'CREATE TABLE IF NOT EXISTS {tablename} ({col_definitions});'

        cursor.execute(create_table_query)
        print(f"Table '{tablename}' created with schema: {col_definitions}")

    create_table_from_df(df, tablename)
    
    # Convert datetime columns to string before inserting
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    df.to_sql(tablename, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()
    print(f"Data Loaded into '{tablename}' table in '{db_name}' SQLite database.")

def get_table_schema(db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    conn.close()
    return schema

def run_sql_query(db_name, query):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute(query)

        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]

        conn.close()

        return results, column_names
    except sqlite3.Error as e:
        print("Following error occurred in execution: ", e)
        return [], []
    
def generate_llm_prompt(table_name, table_schema, question):
    prompt = f"""You are an expert in writing SQL queries for relational databases. 
    You will be provided with a database schema and a natural 
    language question, and your task is to generate an accurate SQL query.
    
    The database has a table named '{table_name}' with the following schema:\n\n"""
    
    prompt += "Columns:\n"

    for col in table_schema:
        column_name = col[1]
        column_type = col[2]
        prompt += f"- {column_name} ({column_type})\n"
    
    prompt += "\nPlease generate a SQL query based on the following natural language question. ONLY return SQL query.\n"
    prompt += f"""QUESTION: {question} \n"""

    return prompt

def generate_sql_query(question, db_name, table_name, llm):
    table_schema = get_table_schema(db_name, table_name)
    llm_prompt = generate_llm_prompt(table_name, table_schema, question)
    try:
        response = llm.invoke(llm_prompt)
        return response
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the Ollama server. Please make sure it's running and try again.")
        return None

def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def find_similar_question(new_question, cache):
    if not cache:
        return None, None

    questions = list(cache.keys())
    vectorizer.fit(questions + [new_question])
    question_vectors = vectorizer.transform(questions)
    new_question_vector = vectorizer.transform([new_question])

    similarities = cosine_similarity(new_question_vector, question_vectors)[0]
    most_similar_index = np.argmax(similarities)
    
    if similarities[most_similar_index] > 0.8:  # Threshold for similarity
        return questions[most_similar_index], cache[questions[most_similar_index]]
    return None, None

def main():
    st.title('Text2SQL')
    
    # Check if Ollama is running
    if not is_ollama_running():
        st.error("Ollama server is not running. Please start the Ollama server and refresh this page.")
        st.info("To start Ollama, open a terminal and run: ollama run llama2:13b")
        return

    # Initialize LLM
    try:
        llm = Ollama(model='llama3.1:8b')
    except Exception as e:
        st.error(f"Error initializing Ollama: {str(e)}")
        return
    
    # File upload
    file = st.file_uploader(label='Upload CSV File', type=['csv'])
    
    if file is not None:
        db_name = 'db.sqlite'
        table_name = 'data_table'
        
        # Load data into SQLite
        csv_to_sqlite(file, db_name, table_name)
        st.success(f"Data loaded into '{table_name}' table in '{db_name}' SQLite database.")
        
        # User input for question
        question = st.text_input(label='Enter your question about the data')
        
        if question:
            # Check cache for similar questions
            similar_question, cached_query = find_similar_question(question, st.session_state.question_cache)
            
            if similar_question:
                st.info(f"Similar question found: {similar_question}")
                st.info(f"Using cached query: {cached_query}")
                query = cached_query
            else:
                # Generate SQL query
                query = generate_sql_query(question, db_name, table_name, llm)
                if query:
                    # Add to cache
                    st.session_state.question_cache[question] = query
            
            if query:
                st.subheader("Generated SQL Query:")
                st.code(query, language='sql')
                
                # Execute query and display results
                results, column_names = run_sql_query(db_name, query)
                if results:
                    st.subheader("Query Results:")
                    df_results = pd.DataFrame(results, columns=column_names)
                    st.dataframe(df_results)
                else:
                    st.warning("No results returned from the query.")
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == '__main__':
    main()