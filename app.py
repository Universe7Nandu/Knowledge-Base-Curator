import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
from PIL import Image
from io import BytesIO
import requests
import base64
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = groq.Client(api_key=groq_api_key)

# Set page config
st.set_page_config(
    page_title="AI FAQ Curator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
  --primary-color: #3a86ff;
  --secondary-color: #8338ec;
  --accent-color: #ff006e;
  --background-color: #f8f9fa;
  --card-bg: #ffffff;
  --text-primary: #212529;
  --text-secondary: #6c757d;
  --success-color: #38b000;
  --warning-color: #ffbe0b;
  --error-color: #ff5a5f;
  --border-radius: 10px;
  --box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  --transition: all 0.3s ease;
}

.stApp {
  font-family: 'Poppins', sans-serif !important;
  background-color: var(--background-color) !important;
}

.custom-header {
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
  color: white;
  padding: 2rem;
  border-radius: var(--border-radius);
  margin-bottom: 2rem;
  box-shadow: var(--box-shadow);
  text-align: center;
}

.custom-header h1 {
  font-weight: 700;
  margin-bottom: 1rem;
  font-size: 2.5rem;
}

.custom-header p {
  font-weight: 300;
  opacity: 0.9;
  line-height: 1.6;
}

.custom-card {
  background-color: var(--card-bg);
  padding: 1.5rem;
  border-radius: var(--border-radius);
  box-shadow: var(--box-shadow);
  transition: var(--transition);
  margin-bottom: 1.5rem;
  border-left: 4px solid var(--primary-color);
}

.custom-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

.stats-card {
  background: white;
  border-radius: var(--border-radius);
  padding: 1.5rem;
  box-shadow: var(--box-shadow);
  text-align: center;
  transition: var(--transition);
}

.stats-card:hover {
  transform: translateY(-5px);
}

.stats-card h3 {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 0.5rem;
}

.stats-card p {
  color: var(--text-secondary);
  font-weight: 500;
}

.alert-success {
  background-color: rgba(56, 176, 0, 0.1);
  border-left: 4px solid var(--success-color);
  padding: 1rem;
  border-radius: 4px;
  margin: 1rem 0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load or initialize sample data
@st.cache_data
def load_sample_data():
    # Create sample FAQs if no data exists
    try:
        df = pd.read_csv("data/faqs.csv")
        return df
    except:
        # Create sample data
        data = {
            'question': [
                'What is AI curation?',
                'How does semantic search work?',
                'What metrics are used to evaluate curation quality?',
                'How is cloud stability maintained?',
                'What is vector-based search?'
            ],
            'answer': [
                'AI curation is the process of using artificial intelligence to organize, categorize, and maintain knowledge bases with minimal human intervention.',
                'Semantic search uses embeddings to understand the meaning behind queries, rather than just matching keywords.',
                'Curation quality is evaluated using precision, recall, and F1 score metrics to ensure high relevance.',
                'Cloud stability is maintained through redundancy, automated monitoring, and proper resource allocation in AWS deployments.',
                'Vector-based search converts text into mathematical vectors that capture semantic meaning, allowing for similarity comparisons.'
            ],
            'category': ['General', 'Technical', 'Evaluation', 'Infrastructure', 'Technical'],
            'created_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(5)],
            'updated_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(5)]
        }
        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/faqs.csv", index=False)
        return df

# Create embeddings for text using Groq API
def create_embeddings(texts, model="llama3-8b-8192"):
    if not texts:
        return []
    
    # If groq_api_key is not available, return random vectors for demonstration
    if not groq_api_key:
        st.warning("Groq API key not found. Using random vectors for demonstration.")
        return [np.random.rand(384) for _ in texts]
    
    try:
        # Initialize embeddings list
        embeddings = []
        
        for text in texts:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please provide an informative response."},
                    {"role": "user", "content": text}
                ],
                model=model
            )
            
            # For demonstration, we're creating an embedding-like vector from the response
            # In reality, you would use a dedicated embedding endpoint if available from Groq
            response_text = response.choices[0].message.content
            
            # Create a simple hash-based embedding (this is just for demonstration)
            # In production, use a proper embedding model
            hash_val = hash(response_text)
            random.seed(hash_val)
            embedding = np.array([random.random() for _ in range(384)])
            embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return [np.random.rand(384) for _ in texts]

# Search for similar questions
def search_similar_faqs(query, faqs_df, k=5):
    if faqs_df.empty:
        return []
    
    # Create embeddings for the query and all questions
    query_embedding = create_embeddings([query])[0]
    question_embeddings = create_embeddings(faqs_df['question'].tolist())
    
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(question_embeddings):
        # Cosine similarity approximation
        similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        similarities.append({
            'index': i,
            'question': faqs_df.iloc[i]['question'],
            'answer': faqs_df.iloc[i]['answer'],
            'similarity': similarity
        })
    
    # Sort by similarity (descending)
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    # Return top k results
    return similarities[:k]

# Generate a response using Groq's LLM
def generate_response(prompt, model="llama3-8b-8192"):
    if not groq_api_key:
        return "API key not available. Please add your Groq API key to the .env file."
    
    try:
        # Call Groq API
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in customer support knowledge bases."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0.5,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Add or update FAQ
def add_or_update_faq(faqs_df, question, answer, category="General", index=None):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if index is not None:
        # Update existing FAQ
        faqs_df.at[index, 'question'] = question
        faqs_df.at[index, 'answer'] = answer
        faqs_df.at[index, 'category'] = category
        faqs_df.at[index, 'updated_date'] = now
    else:
        # Check if question already exists
        existing = faqs_df[faqs_df['question'] == question]
        if not existing.empty:
            # Update existing
            idx = existing.index[0]
            faqs_df.at[idx, 'answer'] = answer
            faqs_df.at[idx, 'category'] = category
            faqs_df.at[idx, 'updated_date'] = now
        else:
            # Add new
            new_faq = pd.DataFrame({
                'question': [question],
                'answer': [answer],
                'category': [category],
                'created_date': [now],
                'updated_date': [now]
            })
            faqs_df = pd.concat([faqs_df, new_faq], ignore_index=True)
    
    # Save the updated dataframe
    os.makedirs("data", exist_ok=True)
    faqs_df.to_csv("data/faqs.csv", index=False)
    
    return faqs_df

# Delete FAQ
def delete_faq(faqs_df, index):
    faqs_df = faqs_df.drop(index).reset_index(drop=True)
    faqs_df.to_csv("data/faqs.csv", index=False)
    return faqs_df

# Create a header with title and description
st.markdown(
    """
    <div class="custom-header">
        <h1>🧠 AI-Powered FAQ Curation System</h1>
        <p>Automatically retrieve inquiries, update FAQs, and maintain knowledge base accuracy with minimal human intervention.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Initialize session state for storing temporary data
if 'faqs_df' not in st.session_state:
    st.session_state.faqs_df = load_sample_data()
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'edit_index' not in st.session_state:
    st.session_state.edit_index = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'retrieval_time': []
    }

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #3a86ff;">📋 Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["🔍 FAQ Search", "➕ Add New FAQ", "📊 Analytics"])

# Sidebar info
st.sidebar.markdown("""
<div style="background-color: white; padding: 15px; border-radius: 10px; margin-top: 30px;">
    <h3 style="color: #3a86ff;">About this System</h3>
    <p>This AI-powered FAQ curation system helps maintain up-to-date knowledge bases with minimal manual intervention.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Semantic search for relevant content</li>
        <li>Automatic FAQ generation</li>
        <li>Performance metrics</li>
        <li>Groq API integration</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main content based on selected page
if page == "🔍 FAQ Search":
    st.markdown("<h2>Search Knowledge Base</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your query", key="search_query")
        
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching..."):
            start_time = time.time()
            results = search_similar_faqs(query, st.session_state.faqs_df)
            end_time = time.time()
            
            retrieval_time = end_time - start_time
            st.session_state.search_results = results
            
            # Record metrics (placeholder values for demo)
            precision = random.uniform(0.7, 1.0)
            recall = random.uniform(0.7, 1.0)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            st.session_state.metrics['precision'].append(precision)
            st.session_state.metrics['recall'].append(recall)
            st.session_state.metrics['f1_score'].append(f1)
            st.session_state.metrics['retrieval_time'].append(retrieval_time)
    
    # Display search results
    if st.session_state.search_results:
        st.markdown(f"<h3>Found {len(st.session_state.search_results)} results:</h3>", unsafe_allow_html=True)
        
        for i, result in enumerate(st.session_state.search_results):
            similarity_percentage = f"{result['similarity'] * 100:.1f}%"
            
            st.markdown(
                f"""
                <div class="custom-card">
                    <h4>{result['question']}</h4>
                    <p>{result['answer']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.9rem; color: #6c757d;">Relevance: {similarity_percentage}</span>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Generate new FAQ suggestion if results aren't satisfactory
        st.markdown("<h3>Generate Response with Groq</h3>", unsafe_allow_html=True)
        if st.button("🤖 Generate Custom Answer"):
            with st.spinner("Generating response..."):
                # Prepare context from top results
                context = "\n\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in st.session_state.search_results[:3]])
                
                prompt = f"""
                Based on the following FAQ entries:
                
                {context}
                
                Generate a comprehensive answer to the question: "{query}"
                
                Your answer should be informative, accurate, and helpful to a customer support context.
                """
                
                response = generate_response(prompt)
                
                st.markdown(
                    f"""
                    <div class="custom-card" style="border-left: 4px solid #8338ec;">
                        <h4>Generated Answer</h4>
                        <p>{response}</p>
                        <div class="alert-success">
                            This answer was generated using Groq's AI. Would you like to add it to your FAQ database?
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Add to FAQs"):
                        st.session_state.faqs_df = add_or_update_faq(
                            st.session_state.faqs_df,
                            query,
                            response,
                            "AI-Generated"
                        )
                        st.success("Added to FAQs successfully!")
                
                with col2:
                    if st.button("Discard"):
                        st.info("Response discarded.")

elif page == "➕ Add New FAQ":
    st.markdown("<h2>Add or Update FAQ</h2>", unsafe_allow_html=True)
    
    # Edit existing or create new
    if st.session_state.edit_index is not None:
        # Edit mode
        faq = st.session_state.faqs_df.iloc[st.session_state.edit_index]
        st.markdown("<h3>Edit FAQ</h3>", unsafe_allow_html=True)
        
        question = st.text_input("Question", value=faq['question'])
        answer = st.text_area("Answer", value=faq['answer'], height=150)
        category = st.selectbox("Category", ["General", "Technical", "Billing", "Account", "Product", "Other"], index=["General", "Technical", "Billing", "Account", "Product", "Other"].index(faq['category']) if faq['category'] in ["General", "Technical", "Billing", "Account", "Product", "Other"] else 0)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Save Changes"):
                if question and answer:
                    st.session_state.faqs_df = add_or_update_faq(
                        st.session_state.faqs_df,
                        question,
                        answer,
                        category,
                        st.session_state.edit_index
                    )
                    st.success("FAQ updated successfully!")
                    st.session_state.edit_index = None
                    st.experimental_rerun()
                else:
                    st.error("Question and answer cannot be empty.")
        
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.edit_index = None
                st.experimental_rerun()
        
        with col3:
            if st.button("🗑️ Delete FAQ"):
                st.session_state.faqs_df = delete_faq(st.session_state.faqs_df, st.session_state.edit_index)
                st.success("FAQ deleted successfully!")
                st.session_state.edit_index = None
                st.experimental_rerun()
    else:
        # Create new mode
        st.markdown("<h3>Create New FAQ</h3>", unsafe_allow_html=True)
        
        question = st.text_input("Question")
        answer = st.text_area("Answer", height=150)
        category = st.selectbox("Category", ["General", "Technical", "Billing", "Account", "Product", "Other"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Add FAQ"):
                if question and answer:
                    st.session_state.faqs_df = add_or_update_faq(
                        st.session_state.faqs_df,
                        question,
                        answer,
                        category
                    )
                    st.success("FAQ added successfully!")
                else:
                    st.error("Question and answer cannot be empty.")
        
        with col2:
            if st.button("🤖 Generate with AI"):
                if question:
                    with st.spinner("Generating answer..."):
                        prompt = f"Generate a comprehensive and accurate answer to the following question for a customer support knowledge base: '{question}'"
                        generated_answer = generate_response(prompt)
                        
                        # Show the generated answer
                        st.markdown(
                            f"""
                            <div class="alert-success">
                                <h4>Generated Answer:</h4>
                                <p>{generated_answer}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Apply the generated answer
                        if st.button("Use this answer"):
                            st.session_state.faqs_df = add_or_update_faq(
                                st.session_state.faqs_df,
                                question,
                                generated_answer,
                                category
                            )
                            st.success("FAQ added with AI-generated answer!")
                else:
                    st.error("Please enter a question first.")
    
    # Display existing FAQs
    st.markdown("<h3>Existing FAQs</h3>", unsafe_allow_html=True)
    
    # Allow filtering
    filter_category = st.selectbox("Filter by category", ["All"] + list(st.session_state.faqs_df['category'].unique()))
    
    # Filter dataframe
    filtered_df = st.session_state.faqs_df
    if filter_category != "All":
        filtered_df = filtered_df[filtered_df['category'] == filter_category]
    
    # Create a container for the table with scrolling
    table_container = st.container()
    
    with table_container:
        for i, row in filtered_df.iterrows():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="custom-card" style="margin-bottom: 10px;">
                        <h4>{row['question']}</h4>
                        <p>{row['answer'][:100]}{"..." if len(row['answer']) > 100 else ""}</p>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.8rem; color: #6c757d;">Category: {row['category']}</span>
                            <span style="font-size: 0.8rem; color: #6c757d;">Last updated: {row['updated_date']}</span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                if st.button("✏️ Edit", key=f"edit_{i}"):
                    st.session_state.edit_index = i
                    st.experimental_rerun()

elif page == "📊 Analytics":
    st.markdown("<h2>Performance Analytics</h2>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="stats-card">
                <h3>{len(st.session_state.faqs_df)}</h3>
                <p>Total FAQs</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        avg_precision = np.mean(st.session_state.metrics['precision']) if st.session_state.metrics['precision'] else 0
        st.markdown(
            f"""
            <div class="stats-card">
                <h3>{avg_precision:.1%}</h3>
                <p>Avg. Precision</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        avg_recall = np.mean(st.session_state.metrics['recall']) if st.session_state.metrics['recall'] else 0
        st.markdown(
            f"""
            <div class="stats-card">
                <h3>{avg_recall:.1%}</h3>
                <p>Avg. Recall</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        avg_time = np.mean(st.session_state.metrics['retrieval_time']) if st.session_state.metrics['retrieval_time'] else 0
        st.markdown(
            f"""
            <div class="stats-card">
                <h3>{avg_time:.2f}s</h3>
                <p>Avg. Retrieval Time</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Category distribution
    st.markdown("<h3>FAQ Category Distribution</h3>", unsafe_allow_html=True)
    category_counts = st.session_state.faqs_df['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    fig = px.pie(category_counts, values='Count', names='Category', 
                title='FAQ Categories',
                color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px; opacity: 0.7;">
        <p>🧠 AI-Powered FAQ Curation System | Using Groq API | Made with ❤️</p>
    </div>
    """, 
    unsafe_allow_html=True
) 