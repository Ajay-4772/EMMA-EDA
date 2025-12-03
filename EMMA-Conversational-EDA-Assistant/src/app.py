import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import sqlite3
import os
from datetime import datetime
from typing import Union, Dict, Any, List
import base64
from io import BytesIO
import requests
import contextlib
import sys
import traceback

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Optional cloud import
try:
    import boto3
    from botocore.exceptions import ClientError
    _AWS_AVAILABLE = True
except Exception:
    _AWS_AVAILABLE = False
try:
    from google.cloud import storage as gcs_storage
    from google.api_core.exceptions import NotFound as GCSNotFound
    _GCS_AVAILABLE = True
except Exception:
    _GCS_AVAILABLE = False
try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
    _AZURE_AVAILABLE = True
except Exception:
    _AZURE_AVAILABLE = False

# Database setup for chat history
def init_database():
    """Initialize SQLite database for chat history"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_title TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            file_name TEXT,
            data_shape TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')
    conn.commit()
    conn.close()

def save_chat_message(session_id: int, user_message: str, bot_response: str):
    """Save a chat message to database"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_messages (session_id, user_message, bot_response)
        VALUES (?, ?, ?)
    ''', (session_id, user_message, str(bot_response)))
    conn.commit()
    conn.close()

def create_chat_session(title: str, file_name: str, data_shape: str) -> int:
    """Create a new chat session and return session ID"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_sessions (session_title, file_name, data_shape)
        VALUES (?, ?, ?)
    ''', (title, file_name, data_shape))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_chat_sessions() -> List[Dict]:
    """Get all chat sessions"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, session_title, timestamp, file_name, data_shape
        FROM chat_sessions
        ORDER BY timestamp DESC
    ''')
    sessions = []
    for row in cursor.fetchall():
        sessions.append({
            'id': row[0],
            'title': row[1],
            'timestamp': row[2],
            'file_name': row[3],
            'data_shape': row[4]
        })
    conn.close()
    return sessions

def get_chat_messages(session_id: int) -> List[Dict]:
    """Get all messages for a specific session"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_message, bot_response, timestamp
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY timestamp
    ''', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.append({
            'user': row[0],
            'bot': row[1],
            'timestamp': row[2]
        })
    conn.close()
    return messages

# Enhanced data loading functions with large file support
def load_data(uploaded_file) -> tuple[Union[pd.DataFrame, str], str]:
    """Load data from uploaded file with enhanced format support and large file handling"""
    file_name = uploaded_file.name.lower()
    file_size = uploaded_file.size
    
    try:
        # Show progress for large files
        if file_size > 100 * 1024 * 1024:  # 100MB
            st.info(f"📁 Loading large file ({file_size / (1024*1024*1024):.1f} GB)... This may take a moment.")
        
        if file_name.endswith('.csv'):
            # Enhanced CSV loading with chunking for large files
            if file_size > 500 * 1024 * 1024:  # 500MB
                # Use chunking for very large files
                chunk_size = int(st.session_state.get('chunk_size', 100000))
                chunks = []
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                # UI elements for streaming/progress
                progress_text = st.empty()
                preview_placeholder = st.empty()
                progress_bar = st.progress(0)
                processed_rows = 0
                start_time = datetime.now()
                est_total = None  # unknown total rows without a pre-pass
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        chunk_iter = pd.read_csv(uploaded_file, encoding=encoding, chunksize=chunk_size)
                        chunk_index = 0
                        for chunk in chunk_iter:
                            chunks.append(chunk)
                            chunk_index += 1
                            processed_rows += len(chunk)
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rows_per_sec = processed_rows / elapsed if elapsed > 0 else 0
                            # Soft progress (cap at 0.95, finish at the end)
                            soft_progress = min(0.95, 0.05 + 0.02 * chunk_index)
                            progress_bar.progress(soft_progress)
                            eta = "calculating..." if rows_per_sec == 0 else f"~{int((file_size/ (1024*1024)) / max(1, (processed_rows/ chunk_size)))} chunks left"
                            progress_text.text(f"🔄 Processed ~{processed_rows:,} rows • Chunk {chunk_index} • {eta}")
                            
                            # Update streaming preview if enabled
                            if st.session_state.get('streaming_enabled'):
                                try:
                                    # Only preview up to the first 1000 rows to keep UI responsive
                                    preview_df = pd.concat(chunks, ignore_index=True)
                                    preview_placeholder.dataframe(preview_df.head(1000), use_container_width=True, height=400)
                                except Exception:
                                    preview_placeholder.dataframe(chunk.head(1000), use_container_width=True, height=400)
                        data = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        progress_text.text("✅ Completed loading all chunks.")
                        return data, "csv"
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any supported encoding")
            else:
                # Standard loading for smaller files
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        data = pd.read_csv(uploaded_file, encoding=encoding)
                        return data, "csv"
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any supported encoding")
                
        elif file_name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, sep='\t')
            return data, "tsv"
        elif file_name.endswith(('.xlsx', '.xls')):
            # Enhanced Excel loading with memory optimization
            if file_size > 100 * 1024 * 1024:  # 100MB
                data = pd.read_excel(uploaded_file, engine='openpyxl', keep_default_na=False)
            else:
                data = pd.read_excel(uploaded_file)
            return data, "excel"
        elif file_name.endswith('.json'):
            data = pd.read_json(uploaded_file)
            return data, "json"
        elif file_name.endswith('.parquet'):
            # Parquet is already optimized; just read directly
            data = pd.read_parquet(uploaded_file)
            return data, "parquet"
        elif file_name.endswith('.txt'):
            data = uploaded_file.read().decode('utf-8')
            return data, "txt"
        elif file_name.endswith('.pdf'):
            # For PDF files, extract text
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text, "pdf"
        else:
            st.error("Unsupported file format")
            return None, None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

def save_results(data: pd.DataFrame, filename: str = "analysis_results.csv"):
    """Save analysis results to file"""
    try:
        data.to_csv(filename, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return False

# Enhanced visualization functions
def create_visualization(data: pd.DataFrame, viz_type: str, x_col: str = None, y_col: str = None, title: str = "Chart"):
    """Create various types of visualizations"""
    if viz_type == "bar":
        fig = px.bar(data, x=x_col, y=y_col, title=title)
    elif viz_type == "pie":
        fig = px.pie(data, values=y_col, names=x_col, title=title)
    elif viz_type == "line":
        fig = px.line(data, x=x_col, y=y_col, title=title)
    elif viz_type == "scatter":
        fig = px.scatter(data, x=x_col, y=y_col, title=title)
    elif viz_type == "box":
        fig = px.box(data, x=x_col, y=y_col, title=title)
    elif viz_type == "histogram":
        fig = px.histogram(data, x=x_col, title=title)
    else:
        fig = px.bar(data, x=x_col, y=y_col, title=title)
    
    fig.update_layout(
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    return fig

def get_download_link(fig, filename: str, file_type: str = "png"):
    """Generate download link for plotly figures"""
    if file_type == "png":
        img_bytes = fig.to_image(format="png")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download PNG</a>'
    elif file_type == "pdf":
        img_bytes = fig.to_image(format="pdf")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF</a>'
    else:
        href = ""
    return href

# GROQ API Integration - Enhanced for Dynamic Analysis
def ask_groq(prompt, model="tngtech/deepseek-r1t2-chimera:free", data=None, api_type="groq"):
    """Send a comprehensive prompt to various APIs for dynamic EDA analysis"""
    
    if api_type == "groq":
        return ask_groq_api(prompt, model, data)
    elif api_type == "ollama":
        return ask_ollama_api(prompt, model, data)
    elif api_type == "openai":
        return ask_openai_api(prompt, model, data)
    else:
        return f"❌ Unsupported API type: {api_type}"

def ask_groq_api(prompt, model="tngtech/deepseek-r1t2-chimera:free", data=None):
    """Send a comprehensive prompt to OpenRouter API for dynamic EDA analysis"""
    api_key = ""  # Replace with your OpenRouter API key
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    if data is not None and isinstance(data, pd.DataFrame):
        # Get comprehensive dataset information
        col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(data.columns, data.dtypes)])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Send a sample of the dataset for analysis to avoid token limits
        if len(data) > 50:
            # For large datasets, send a sample + summary statistics
            sample_data = data.head(20).to_markdown(index=False)
            data_summary = f"""
DATASET SUMMARY:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Sample shown: First 20 rows
- Data types: {dict(data.dtypes)}
- Missing values: {dict(data.isnull().sum())}
- Numeric columns: {data.select_dtypes(include=[np.number]).columns.tolist()}
- Categorical columns: {data.select_dtypes(include=['object']).columns.tolist()}

SAMPLE DATA (First 20 rows):
{sample_data}
"""
            full_data = data_summary
        else:
            # For smaller datasets, send the complete data
            full_data = data.to_markdown(index=False)
        
        # Enhanced system prompt for conversational EDA like ChatGPT
        system_prompt = f"""You are EMMA, a friendly and intelligent data analysis assistant. Think of yourself as ChatGPT but specialized in data analysis. Be conversational, helpful, and engaging.

CORE PRINCIPLES:
1. **Conversational Style**: Talk like ChatGPT - friendly, clear, and engaging. Use natural language and be helpful.
2. **Smart Analysis**: Analyze the data dynamically and provide insights based on actual values.
3. **Context Awareness**: Remember previous questions and build on the conversation.
4. **Helpful Responses**: Provide actionable insights and explain things clearly.
5. **Visualization Only When Asked**: Only create charts when explicitly requested with words like "pie chart", "bar chart", "heatmap", "visualize", etc.
6. **No Automatic Charts**: Don't generate charts for general questions like "What's the average?" or "Who is the highest?" - just answer conversationally.
7. **No Code Explanations**: When creating visualizations, generate ONLY executable code, no text explanations or insights.

DATASET CONTEXT:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Columns: {col_info}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

COMPLETE DATASET (all {len(data)} rows):
{full_data}

ANALYSIS CAPABILITIES:
🔷 **Statistical Analysis**: Mean, median, mode, std dev, variance, percentiles, IQR
🔷 **Data Quality**: Missing values, duplicates, data types, outliers
🔷 **Distribution Analysis**: Histograms, box plots, density plots, skewness, kurtosis
🔷 **Correlation Analysis**: Pearson, Spearman correlations, heatmaps
🔷 **Grouping & Aggregation**: Group by analysis, pivot tables, aggregations
🔷 **Filtering & Segmentation**: Conditional filtering, data segmentation
🔷 **Ranking & Sorting**: Top-N analysis, ranking, performance metrics
🔷 **Visualization Guidance**: Chart type recommendations, plotting strategies
🔷 **Feature Engineering**: New column creation, data transformations
🔷 **Outlier Detection**: IQR method, z-score analysis, specific identification
🔷 **Categorical Analysis**: Frequency analysis, category comparisons
🔷 **Time Series Analysis**: Trends, seasonality, temporal patterns
🔷 **ML Context**: Feature importance, target analysis, model insights

RESPONSE FORMAT:
- Start with a clear, conversational answer to the user's question
- Use markdown formatting for better readability
- Include specific numbers, percentages, and data points
- Provide context and interpretation of findings
- Use bullet points and sections for complex analyses
- Be helpful and educational in your explanations

VISUALIZATION GUIDANCE:
When users ask for visualizations, suggest the most appropriate chart type:
- **Distributions**: Histograms, box plots, violin plots, KDE plots
- **Relationships**: Scatter plots, line charts, correlation heatmaps
- **Categories**: Bar charts, pie charts, count plots, strip plots
- **Comparisons**: Grouped bar charts, side-by-side plots
- **Time Series**: Line charts, area charts, trend plots
- **Correlations**: Heatmaps, correlation matrices, pair plots

CRITICAL: When asked for visualizations, you MUST provide executable Python code using matplotlib (plt), NOT text descriptions of charts.

Remember: You are thinking and analyzing in real-time. Every response should be unique and based on the current data analysis. Provide specific insights, actual numbers, and actionable recommendations."""

        user_content = f"User question: {prompt}"
    else:
        system_prompt = """You are EMMA, an expert EDA assistant. Provide helpful guidance for data analysis questions."""
        user_content = f"User question: {prompt}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"[Groq API error: {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[Groq connection error: {str(e)}]"

def ask_ollama_api(prompt, model="mistral", data=None):
    """Send a comprehensive prompt to local Ollama API for dynamic EDA analysis"""
    url = "http://localhost:11434/api/generate"
    
    if data is not None and isinstance(data, pd.DataFrame):
        # Get comprehensive dataset information
        col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(data.columns, data.dtypes)])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Send a sample of the dataset for analysis to avoid token limits
        if len(data) > 50:
            # For large datasets, send a sample + summary statistics
            sample_data = data.head(20).to_markdown(index=False)
            data_summary = f"""
DATASET SUMMARY:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Sample shown: First 20 rows
- Data types: {dict(data.dtypes)}
- Missing values: {dict(data.isnull().sum())}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

SAMPLE DATA (First 20 rows):
{sample_data}
"""
            full_data = data_summary
        else:
            # For smaller datasets, send the complete data
            full_data = data.to_markdown(index=False)
        
        # Enhanced system prompt for comprehensive EDA with code generation
        system_prompt = f"""You are EMMA, an expert Exploratory Data Analysis (EDA) assistant powered by Ollama. You are designed to think dynamically and provide comprehensive, accurate analysis based on the actual data provided.

CORE PRINCIPLES:
1. **Dynamic Thinking**: Never use predefined or hardcoded responses. Analyze the data in real-time and provide unique insights.
2. **Complete Analysis**: Always analyze the ENTIRE dataset, not just samples or subsets.
3. **Data-Driven Responses**: Base all answers on actual values, statistics, and patterns in the data.
4. **ChatGPT-Style Communication**: Be conversational, clear, and helpful. Use natural language with proper formatting.
5. **Professional EDA**: Provide statistical rigor with practical insights.
6. **Code Generation**: When asked for visualizations, ALWAYS provide executable Python code, not text descriptions.

DATASET CONTEXT:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Columns: {col_info}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

COMPLETE DATASET (all {len(data)} rows):
{full_data}

ANALYSIS CAPABILITIES:
🔷 **Statistical Analysis**: Mean, median, mode, std dev, variance, percentiles, IQR
🔷 **Data Quality**: Missing values, duplicates, data types, outliers
🔷 **Distribution Analysis**: Histograms, box plots, density plots, skewness, kurtosis
🔷 **Correlation Analysis**: Pearson, Spearman correlations, heatmaps
🔷 **Grouping & Aggregation**: Group by analysis, pivot tables, aggregations
🔷 **Filtering & Segmentation**: Conditional filtering, data segmentation
🔷 **Ranking & Sorting**: Top-N analysis, ranking, performance metrics
🔷 **Visualization Guidance**: Chart type recommendations, plotting strategies
🔷 **Feature Engineering**: New column creation, data transformations
🔷 **Outlier Detection**: IQR method, z-score analysis, specific identification
🔷 **Categorical Analysis**: Frequency analysis, category comparisons
🔷 **Time Series Analysis**: Trends, seasonality, temporal patterns
🔷 **ML Context**: Feature importance, target analysis, model insights

RESPONSE FORMAT:
- Start with a clear, conversational answer to the user's question
- Use markdown formatting for better readability
- Include specific numbers, percentages, and data points
- Provide context and interpretation of findings
- Use bullet points and sections for complex analyses
- Be helpful and educational in your explanations

VISUALIZATION GUIDANCE:
When users ask for visualizations, suggest the most appropriate chart type:
- **Distributions**: Histograms, box plots, violin plots, KDE plots
- **Relationships**: Scatter plots, line charts, correlation heatmaps
- **Categories**: Bar charts, pie charts, count plots, strip plots
- **Comparisons**: Grouped bar charts, side-by-side plots
- **Time Series**: Line charts, area charts, trend plots
- **Correlations**: Heatmaps, correlation matrices, pair plots

CRITICAL: When asked for visualizations, you MUST provide executable Python code using matplotlib (plt), NOT text descriptions of charts.

Remember: You are thinking and analyzing in real-time. Every response should be unique and based on the current data analysis. Provide specific insights, actual numbers, and actionable recommendations."""

        user_content = f"User question: {prompt}"
    else:
        system_prompt = """You are EMMA, an expert EDA assistant. Provide helpful guidance for data analysis questions."""
        user_content = f"User question: {prompt}"
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_content}",
        "stream": False,
        "options": {
            "num_predict": 4000,  # Increased for more detailed responses
            "temperature": 0.3,   # Slightly higher for more conversational tone
            "top_k": 20,          # More token variety
            "top_p": 0.9          # Nucleus sampling
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)  # Increased timeout
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "[No response from model]")
        else:
            return f"[Ollama API error: {response.status_code} - {response.text}]"
    except requests.exceptions.Timeout:
        return "⏰ Sorry, I'm taking a bit longer than expected. Please try again!"
    except Exception as e:
        return f"❌ Ollama connection error: {str(e)}"

def ask_openai_api(prompt, model="gpt-3.5-turbo", data=None):
    """Send a comprehensive prompt to OpenAI API for dynamic EDA analysis"""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    url = "https://api.openai.com/v1/chat/completions"
    
    if data is not None and isinstance(data, pd.DataFrame):
        # Get comprehensive dataset information
        col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(data.columns, data.dtypes)])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Send a sample of the dataset for analysis to avoid token limits
        if len(data) > 50:
            # For large datasets, send a sample + summary statistics
            sample_data = data.head(20).to_markdown(index=False)
            data_summary = f"""
DATASET SUMMARY:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Sample shown: First 20 rows
- Data types: {dict(data.dtypes)}
- Missing values: {dict(data.isnull().sum())}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

SAMPLE DATA (First 20 rows):
{sample_data}
"""
            full_data = data_summary
        else:
            # For smaller datasets, send the complete data
            full_data = data.to_markdown(index=False)
        
        # Enhanced system prompt for comprehensive EDA with code generation
        system_prompt = f"""You are EMMA, an expert Exploratory Data Analysis (EDA) assistant powered by OpenAI. You are designed to think dynamically and provide comprehensive, accurate analysis based on the actual data provided.

CORE PRINCIPLES:
1. **Dynamic Thinking**: Never use predefined or hardcoded responses. Analyze the data in real-time and provide unique insights.
2. **Complete Analysis**: Always analyze the ENTIRE dataset, not just samples or subsets.
3. **Data-Driven Responses**: Base all answers on actual values, statistics, and patterns in the data.
4. **ChatGPT-Style Communication**: Be conversational, clear, and helpful. Use natural language with proper formatting.
5. **Professional EDA**: Provide statistical rigor with practical insights.
6. **Code Generation**: When asked for visualizations, ALWAYS provide executable Python code, not text descriptions.

DATASET CONTEXT:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}
- Columns: {col_info}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

COMPLETE DATASET (all {len(data)} rows):
{full_data}

ANALYSIS CAPABILITIES:
🔷 **Statistical Analysis**: Mean, median, mode, std dev, variance, percentiles, IQR
🔷 **Data Quality**: Missing values, duplicates, data types, outliers
🔷 **Distribution Analysis**: Histograms, box plots, density plots, skewness, kurtosis
🔷 **Correlation Analysis**: Pearson, Spearman correlations, heatmaps
🔷 **Grouping & Aggregation**: Group by analysis, pivot tables, aggregations
🔷 **Filtering & Segmentation**: Conditional filtering, data segmentation
🔷 **Ranking & Sorting**: Top-N analysis, ranking, performance metrics
🔷 **Visualization Guidance**: Chart type recommendations, plotting strategies
🔷 **Feature Engineering**: New column creation, data transformations
🔷 **Outlier Detection**: IQR method, z-score analysis, specific identification
🔷 **Categorical Analysis**: Frequency analysis, category comparisons
🔷 **Time Series Analysis**: Trends, seasonality, temporal patterns
🔷 **ML Context**: Feature importance, target analysis, model insights

RESPONSE FORMAT:
- Start with a clear, conversational answer to the user's question
- Use markdown formatting for better readability
- Include specific numbers, percentages, and data points
- Provide context and interpretation of findings
- Use bullet points and sections for complex analyses
- Be helpful and educational in your explanations

VISUALIZATION GUIDANCE:
When users ask for visualizations, suggest the most appropriate chart type:
- **Distributions**: Histograms, box plots, violin plots, KDE plots
- **Relationships**: Scatter plots, line charts, correlation heatmaps
- **Categories**: Bar charts, pie charts, count plots, strip plots
- **Comparisons**: Grouped bar charts, side-by-side plots
- **Time Series**: Line charts, area charts, trend plots
- **Correlations**: Heatmaps, correlation matrices, pair plots

CRITICAL: When asked for visualizations, you MUST provide executable Python code using matplotlib (plt), NOT text descriptions of charts.

Remember: You are thinking and analyzing in real-time. Every response should be unique and based on the current data analysis. Provide specific insights, actual numbers, and actionable recommendations."""

        user_content = f"User question: {prompt}"
    else:
        system_prompt = """You are EMMA, an expert EDA assistant. Provide helpful guidance for data analysis questions."""
        user_content = f"User question: {prompt}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"[OpenAI API error: {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[OpenAI connection error: {str(e)}]"

def extract_code_from_response(response_text):
    """Extract and clean Python code from LLM response with enhanced robustness"""
    import re
    
    # Look for code blocks with more flexible patterns
    code_blocks = re.findall(r"```(?:python)?\n([\s\S]+?)```", response_text)
    if code_blocks:
        code = code_blocks[0]
    else:
        # Look for code blocks without language specification
        code_blocks = re.findall(r"```\n([\s\S]+?)```", response_text)
        if code_blocks:
            code = code_blocks[0]
        else:
            # Look for lines that contain Python code patterns
            lines = response_text.splitlines()
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                # Check if this looks like Python code
                if (line.startswith(('import', 'from', 'plt.', 'data.', 'fig', 'sns.', 'px.')) or 
                    line.startswith((' ', '\t')) or
                    '=' in line or
                    '(' in line or
                    line.endswith(')') or
                    'plt.' in line or
                    'data.' in line or
                    'fig.' in line or
                    'sns.' in line or
                    'px.' in line or
                    'seaborn.' in line):
                    code_lines.append(line)
                    in_code_block = True
                elif in_code_block and line == '':
                    code_lines.append(line)
                elif in_code_block and not line.startswith((' ', '\t')) and not any(keyword in line for keyword in ['plt.', 'data.', 'fig', 'import', 'sns.', 'px.']):
                    break
            
            if code_lines:
                code = '\n'.join(code_lines)
            else:
                # If no code found, try to generate basic visualization code based on the request
                return generate_fallback_code(response_text)
    
    # Clean the code
    cleaned_lines = []
    for line in code.splitlines():
        line = line.strip()
        # Skip imports, comments, and empty lines
        if (not line.startswith('import') and 
            not line.startswith('from') and 
            not line.startswith('#') and 
            not line.startswith('"""') and
            not line.startswith("'''") and
            line != '' and
            not line.startswith('fig.show()') and
            not line.startswith('plt.show()') and
            not line.startswith('print(') and
            not line.startswith('display(')):
            cleaned_lines.append(line)
    
    cleaned_code = '\n'.join(cleaned_lines)
    
    # Fix common issues
    cleaned_code = cleaned_code.replace('df.', 'data.')
    cleaned_code = cleaned_code.replace('df[', 'data[')
    cleaned_code = cleaned_code.replace('df(', 'data(')
    cleaned_code = cleaned_code.replace('.show()', '')
    
    # Fix plotly references
    cleaned_code = cleaned_code.replace('plotly.express', 'px')
    cleaned_code = cleaned_code.replace('plotly.graph_objects', 'go')
    cleaned_code = cleaned_code.replace('plotly.', 'px.')
    
    # Fix correlation issues
    if 'data.corr()' in cleaned_code and 'select_dtypes' not in cleaned_code:
        cleaned_code = cleaned_code.replace('data.corr()', 'data.select_dtypes(include=[np.number]).corr()')
    
    # Ensure proper variable assignments and figure creation
    lines = cleaned_code.split('\n')
    fixed_lines = []
    has_fig_assignment = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Fix common patterns
        if 'data.nlargest(5, \'salary\')' in line and '=' not in line:
            fixed_lines.append('top_earners = data.nlargest(5, \'salary\')')
        elif 'px.bar(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.bar(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.scatter(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.hist(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        elif 'plt.plot(' in line and 'fig =' not in line:
            fixed_lines.append(f'fig = {line}')
            has_fig_assignment = True
        else:
            fixed_lines.append(line)
    
    cleaned_code = '\n'.join(fixed_lines)
    
    # Ensure we have a figure assignment at the end
    if not has_fig_assignment and not cleaned_code.endswith('fig = plt.gcf()'):
        cleaned_code += '\nfig = plt.gcf()'
    
    # Debug output
    print(f"DEBUG - Original code: {code[:200]}...")
    print(f"DEBUG - Cleaned code: {cleaned_code}")
    
    return cleaned_code

def generate_fallback_code(response_text):
    """Generate basic visualization code when LLM doesn't provide code"""
    import re
    
    # Extract keywords from the response to determine chart type
    response_lower = response_text.lower()
    
    # Determine chart type based on keywords
    if any(word in response_lower for word in ['heatmap', 'heat map', 'correlation matrix']):
        return """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12, 8))
if 'City' in data.columns and 'Product' in data.columns and 'Total_Amount' in data.columns:
    # Group data by City and Product, calculate average sales
    city_product_sales = data.groupby(['City', 'Product'])['Total_Amount'].mean().unstack()
    sns.heatmap(city_product_sales, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title("Average Sales by City and Product")
    plt.xlabel("Product")
    plt.ylabel("City")
    plt.xticks(rotation=45)
elif 'Total_Amount' in data.columns and len(data.select_dtypes(include=[np.number]).columns) > 1:
    # Create correlation heatmap for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
else:
    # Default heatmap
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
    else:
        plt.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Data Visualization')
plt.tight_layout()
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['pie', 'pie chart', 'proportion', 'percentage']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))
if 'Total_Amount' in data.columns and 'Category' in data.columns:
    # Group by category and sum amounts
    category_sales = data.groupby('Category')['Total_Amount'].sum()
    plt.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
    plt.title('Sales Proportion by Category')
elif 'Total_Amount' in data.columns and 'Product' in data.columns:
    # Group by product and sum amounts
    product_sales = data.groupby('Product')['Total_Amount'].sum()
    plt.pie(product_sales.values, labels=product_sales.index, autopct='%1.1f%%')
    plt.title('Sales Proportion by Product')
elif 'salary' in data.columns and 'name' in data.columns:
    plt.pie(data['salary'], labels=data['name'], autopct='%1.1f%%')
    plt.title('Salary Distribution by Name')
elif 'age' in data.columns and 'name' in data.columns:
    plt.pie(data['age'], labels=data['name'], autopct='%1.1f%%')
    plt.title('Age Distribution by Name')
else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.pie(data[numeric_cols[0]], labels=data.index, autopct='%1.1f%%')
        plt.title(f'{numeric_cols[0]} Distribution')
plt.axis('equal')
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['bar', 'bar chart', 'count', 'highest', 'top', 'revenue', 'sales']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))
if 'Total_Amount' in data.columns and 'Category' in data.columns:
    # Group by category and sum amounts
    category_sales = data.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)
    plt.bar(category_sales.index, category_sales.values)
    plt.title('Total Sales by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
elif 'Total_Amount' in data.columns and 'Product' in data.columns:
    # Group by product and sum amounts
    product_sales = data.groupby('Product')['Total_Amount'].sum().sort_values(ascending=False)
    plt.bar(product_sales.index, product_sales.values)
    plt.title('Total Sales by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
elif 'Total_Amount' in data.columns and 'Customer_Name' in data.columns:
    # Group by customer and sum amounts
    customer_sales = data.groupby('Customer_Name')['Total_Amount'].sum().sort_values(ascending=False)
    plt.bar(customer_sales.index, customer_sales.values)
    plt.title('Total Sales by Customer')
    plt.xlabel('Customer')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
elif 'salary' in data.columns:
    plt.bar(range(len(data)), data['salary'])
    plt.title('Salary Distribution')
    plt.xlabel('Employee Index')
    plt.ylabel('Salary')
elif 'age' in data.columns:
    plt.bar(range(len(data)), data['age'])
    plt.title('Age Distribution')
    plt.xlabel('Employee Index')
    plt.ylabel('Age')
else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.bar(range(len(data)), data[numeric_cols[0]])
        plt.title(f'{numeric_cols[0]} Distribution')
        plt.xlabel('Index')
        plt.ylabel(numeric_cols[0])
plt.tight_layout()
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['histogram', 'distribution', 'hist', 'average', 'quantity', 'items']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
if 'Quantity' in data.columns:
    plt.hist(data['Quantity'], bins=range(1, data['Quantity'].max()+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Quantity Distribution per Transaction')
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')
    plt.xticks(range(1, data['Quantity'].max()+1))
elif 'Total_Amount' in data.columns:
    plt.hist(data['Total_Amount'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Total Amount Distribution')
    plt.xlabel('Total Amount')
    plt.ylabel('Frequency')
elif 'Unit_Price' in data.columns:
    plt.hist(data['Unit_Price'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Unit Price Distribution')
    plt.xlabel('Unit Price')
    plt.ylabel('Frequency')
elif 'salary' in data.columns:
    plt.hist(data['salary'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Salary Distribution Histogram')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
elif 'age' in data.columns:
    plt.hist(data['age'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Age Distribution Histogram')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plt.hist(data[numeric_cols[0]], bins=10, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'{numeric_cols[0]} Distribution Histogram')
        plt.xlabel(numeric_cols[0])
        plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
fig = plt.gcf()"""
    
    elif any(word in response_lower for word in ['scatter', 'correlation', 'relationship']):
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) >= 2:
    plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]], alpha=0.7)
    plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
elif 'salary' in data.columns and 'age' in data.columns:
    plt.scatter(data['age'], data['salary'], alpha=0.7)
    plt.title('Age vs Salary')
    plt.xlabel('Age')
    plt.ylabel('Salary')
else:
    plt.text(0.5, 0.5, 'Insufficient numeric data for scatter plot', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Data Visualization')
plt.grid(True, alpha=0.3)
fig = plt.gcf()"""
    
    else:
        # Default to a simple bar chart
        return """import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    plt.bar(range(len(data)), data[numeric_cols[0]])
    plt.title(f'{numeric_cols[0]} Distribution')
    plt.xlabel('Index')
    plt.ylabel(numeric_cols[0])
else:
    plt.text(0.5, 0.5, 'No numeric data available for visualization', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Data Overview')
plt.tight_layout()
fig = plt.gcf()"""

# Theme-aware CSS
def get_theme_css():
    """Get CSS for clean ChatGPT-like styling"""
    if st.session_state.dark_mode:
        return '''
        <style>
        /* Dark Theme - Clean like ChatGPT */
        .stApp {
            background-color: #343541 !important;
            color: #ffffff !important;
        }
        
        .main .block-container {
            background-color: #343541 !important;
            color: #ffffff !important;
            padding-top: 1rem;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #ffffff !important;
            font-weight: 400;
        }
        
        .stMarkdown label, .stTextInput label, .stFileUploader label {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 1em !important;
        }
        
        .stButton>button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background: #0d8a6f;
        }
        
        .stTextInput>div>div>input {
            background: #40414f !important;
            color: #ffffff !important;
            border: 1px solid #565869 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-size: 0.9em !important;
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #10a37f !important;
            outline: none !important;
        }
        
        .stFileUploader {
            background: #40414f !important;
            border: 2px dashed #565869 !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        
        .chat-bubble {
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 100%;
            font-size: 0.9em;
            background: #444654;
            color: #ffffff;
            border: none;
        }
        .user-bubble {
            background: #343541;
            color: #ffffff;
            margin-left: 0;
            font-weight: 400;
            border: 1px solid #565869;
        }
        .bot-bubble {
            background: #444654;
            color: #ffffff;
            margin-right: 0;
            border: none;
            font-weight: 400;
        }
        
        .stDataFrame {
            background: #40414f !important;
            color: #ffffff !important;
            border-radius: 6px;
            border: 1px solid #565869;
        }
        
        /* Form submit button styling */
        .stFormSubmitButton > button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
            width: 100%;
        }
        .stFormSubmitButton > button:hover {
            background: #0d8a6f;
        }
        
        /* Success and error messages */
        .stSuccess {
            background: #0c4a6e !important;
            border: 1px solid #0ea5e9 !important;
            color: #7dd3fc !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        
        .stError {
            background: #7f1d1d !important;
            border: 1px solid #f87171 !important;
            color: #fca5a5 !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        </style>
        '''
    else:
        return '''
        <style>
        /* Light Theme - Clean like ChatGPT */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        .main .block-container {
            background-color: #ffffff !important;
            color: #000000 !important;
            padding-top: 1rem;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #000000 !important;
            font-weight: 600;
        }
        
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #000000 !important;
            font-weight: 400;
        }
        
        .stMarkdown label, .stTextInput label, .stFileUploader label {
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 1em !important;
        }
        
        .stButton>button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background: #0d8a6f;
        }
        
        .stTextInput>div>div>input {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #e5e5e5 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-size: 0.9em !important;
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #10a37f !important;
            outline: none !important;
        }
        
        .stFileUploader {
            background: #f7f7f8 !important;
            border: 2px dashed #e5e5e5 !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        
        .chat-bubble {
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 100%;
            font-size: 0.9em;
            background: #ffffff;
            color: #000000;
            border: 1px solid #e5e5e5;
        }
        .user-bubble {
            background: #f7f7f8;
            color: #000000;
            margin-left: 0;
            font-weight: 400;
            border: 1px solid #e5e5e5;
        }
        .bot-bubble {
            background: #ffffff;
            color: #000000;
            margin-right: 0;
            border: 1px solid #e5e5e5;
            font-weight: 400;
        }
        
        .stDataFrame {
            background: #ffffff !important;
            color: #000000 !important;
            border-radius: 6px;
            border: 1px solid #e5e5e5;
        }
        
        /* Form submit button styling */
        .stFormSubmitButton > button {
            border-radius: 6px;
            border: none;
            padding: 0.5em 1em;
            font-size: 0.9em;
            background: #10a37f;
            color: #ffffff;
            font-weight: 500;
            transition: background-color 0.2s ease;
            width: 100%;
        }
        .stFormSubmitButton > button:hover {
            background: #0d8a6f;
        }
        
        /* Success and error messages */
        .stSuccess {
            background: #f0f9ff !important;
            border: 1px solid #bae6fd !important;
            color: #0369a1 !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        
        .stError {
            background: #fef2f2 !important;
            border: 1px solid #fecaca !important;
            color: #dc2626 !important;
            border-radius: 6px;
            padding: 0.75rem;
        }
        </style>
        '''

def _load_from_s3(bucket: str, key: str) -> tuple[Union[pd.DataFrame, str], str]:
    """Load an object from S3 into a DataFrame or text depending on extension."""
    if not _AWS_AVAILABLE:
        st.error("boto3 not installed. Install boto3 or use Local File.")
        return None, None
    try:
        s3 = boto3.client("s3")
        with st.spinner("Downloading from S3..."):
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
        name = key.lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body)), "csv"
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(body), sep="\t"), "tsv"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(body)), "excel"
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(body)), "json"
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body)), "parquet"
        if name.endswith(".txt"):
            return body.decode("utf-8"), "txt"
        st.error("Unsupported S3 object type")
        return None, None
    except ClientError as e:
        st.error(f"S3 error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load from S3: {e}")
        return None, None

def _load_from_gcs(bucket: str, blob_name: str) -> tuple[Union[pd.DataFrame, str], str]:
    """Load an object from Google Cloud Storage into a DataFrame or text."""
    if not _GCS_AVAILABLE:
        st.error("google-cloud-storage not installed. Install it or use Local File.")
        return None, None
    try:
        client = gcs_storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        with st.spinner("Downloading from GCS..."):
            body = blob.download_as_bytes()
        name = blob_name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body)), "csv"
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(body), sep="\t"), "tsv"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(body)), "excel"
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(body)), "json"
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body)), "parquet"
        if name.endswith(".txt"):
            return body.decode("utf-8"), "txt"
        st.error("Unsupported GCS object type")
        return None, None
    except GCSNotFound as e:
        st.error(f"GCS error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load from GCS: {e}")
        return None, None

def _load_from_azure(container: str, blob_name: str, connection_string: str | None = None) -> tuple[Union[pd.DataFrame, str], str]:
    """Load an object from Azure Blob Storage into a DataFrame or text."""
    if not _AZURE_AVAILABLE:
        st.error("azure-storage-blob not installed. Install it or use Local File.")
        return None, None
    try:
        if not connection_string:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if not connection_string:
            st.error("Set AZURE_STORAGE_CONNECTION_STRING env var or provide it here.")
            return None, None
        service = BlobServiceClient.from_connection_string(connection_string)
        container_client = service.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        with st.spinner("Downloading from Azure Blob..."):
            body = blob_client.download_blob().readall()
        name = blob_name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(body)), "csv"
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(body), sep="\t"), "tsv"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(body)), "excel"
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(body)), "json"
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body)), "parquet"
        if name.endswith(".txt"):
            return body.decode("utf-8"), "txt"
        st.error("Unsupported Azure blob type")
        return None, None
    except ResourceNotFoundError as e:
        st.error(f"Azure error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Failed to load from Azure: {e}")
        return None, None

# Ensure fixed model silently
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "tngtech/deepseek-r1t2-chimera:free"

# Utility: clear all chat history
def _clear_all_history():
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_messages')
        cursor.execute('DELETE FROM chat_sessions')
        conn.commit()
        conn.close()
    except Exception:
        pass
    # Clear session
    st.session_state.pop('chat_history', None)
    st.session_state.pop('current_session_id', None)

def main():
    # Initialize database
    init_database()
    
    # Theme toggle in sidebar
    with st.sidebar:
        st.title("🎛️ EMMA Settings")
        
        # Theme toggle
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.markdown("---")
        
        # Chat history section
        st.subheader("📚 Chat History")
        
        # Clear all history
        if st.button("🧹 Clear All History"):
            _clear_all_history()
            st.success("History cleared.")
            st.rerun()
        
        # Search functionality
        search_term = st.text_input("🔍 Search chats:", placeholder="Enter keywords...")
        
        # Get chat sessions
        sessions = get_chat_sessions()
        
        if search_term:
            # Filter sessions based on search term
            filtered_sessions = []
            for session in sessions:
                if (search_term.lower() in session['title'].lower() or 
                    search_term.lower() in session['file_name'].lower()):
                    filtered_sessions.append(session)
            sessions = filtered_sessions
        
        # Display chat sessions
        for session in sessions:
            with st.expander(f"📄 {session['title']} ({session['timestamp'][:10]})"):
                st.write(f"**File:** {session['file_name']}")
                st.write(f"**Data Shape:** {session['data_shape']}")
                if st.button(f"Load Session {session['id']}", key=f"load_{session['id']}"):
                    st.session_state.current_session_id = session['id']
                    st.session_state.chat_history = get_chat_messages(session['id'])
                    st.rerun()
    
    # Apply theme CSS
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    st.title("🤖 Meet EMMA: Your EDA Assistant")
    st.write("Hi, I'm EMMA! Upload your data and ask questions in natural language. I'll help you explore, visualize, and understand your data with smart suggestions and insights! ✨")

    # Session controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Chat & Data", key="clear_chat"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("Download Session", key="download_session"):
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                # Create session summary
                summary = "EMMA Chat Session Summary\n" + "="*30 + "\n\n"
                for i, chat in enumerate(st.session_state.chat_history, 1):
                    summary += f"Q{i}: {chat['user']}\n"
                    summary += f"A{i}: {chat['bot']}\n\n"
                
                st.download_button(
                    label="📥 Download Session Summary",
                    data=summary,
                    file_name=f"emma_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    with col3:
        if st.button("Export Data", key="export_data"):
            if 'data' in st.session_state and isinstance(st.session_state['data'], pd.DataFrame):
                csv = st.session_state['data'].to_csv(index=False)
                st.download_button(
                    label="📊 Download Dataset",
                    data=csv,
                    file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # File upload method controls
    st.markdown("---")
    st.subheader("📁 Data Ingestion")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        upload_method = st.radio(
            "Choose upload method:",
            ["Local File", "Cloud Storage", "Direct URL"],
            horizontal=True,
            help="Select how you want to bring data into EMMA"
        )
    with col_u2:
        streaming_enabled = st.checkbox("🔄 Streaming mode", help="Process data incrementally (for very large files)")
    if 'streaming_enabled' not in st.session_state:
        st.session_state.streaming_enabled = False
    st.session_state.streaming_enabled = streaming_enabled

    # Advanced ingestion controls
    adv_col1, adv_col2 = st.columns([1, 1])
    with adv_col1:
        chunk_size = st.slider(
            "Chunk size (rows per chunk for very large CSVs)",
            min_value=10000, max_value=500000, step=10000, value=int(st.session_state.get('chunk_size', 100000)),
            help="Applied when file > 500MB"
        )
    with adv_col2:
        st.caption("Tip: Larger chunks are faster but use more memory.")
    st.session_state['chunk_size'] = chunk_size

    # File upload with large file support
    if upload_method == "Local File":
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, JSON, TSV, Parquet, PDF, or TXT (Supports up to 1GB files)",
            type=["csv", "xlsx", "xls", "json", "tsv", "parquet", "pdf", "txt"],
            help="Large files are processed with chunking automatically"
        )
    elif upload_method == "Cloud Storage":
        platform = st.selectbox("Cloud Platform", ["AWS S3", "Google Cloud Storage", "Azure Blob"]) 
        if platform == "AWS S3":
            c1, c2 = st.columns(2)
            with c1:
                s3_bucket = st.text_input("S3 Bucket")
            with c2:
                s3_key = st.text_input("S3 Key (path/to/file.csv)")
            if st.button("Load from S3"):
                if s3_bucket and s3_key:
                    data, ftype = _load_from_s3(s3_bucket, s3_key)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['file_type'] = ftype
                        st.success(f"✅ Loaded from S3: {len(data) if isinstance(data, pd.DataFrame) else 'text'}")
                else:
                    st.warning("Enter both bucket and key.")
        elif platform == "Google Cloud Storage":
            c1, c2 = st.columns(2)
            with c1:
                gcs_bucket = st.text_input("GCS Bucket")
            with c2:
                gcs_blob = st.text_input("Blob (path/to/file.csv)")
            if st.button("Load from GCS"):
                if gcs_bucket and gcs_blob:
                    data, ftype = _load_from_gcs(gcs_bucket, gcs_blob)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['file_type'] = ftype
                        st.success(f"✅ Loaded from GCS: {len(data) if isinstance(data, pd.DataFrame) else 'text'}")
                else:
                    st.warning("Enter both bucket and blob name.")
        else:  # Azure
            c1, c2 = st.columns(2)
            with c1:
                az_container = st.text_input("Azure Container")
            with c2:
                az_blob = st.text_input("Blob (path/to/file.csv)")
            az_conn = st.text_input("Connection String (optional)", type="password")
            if st.button("Load from Azure"):
                if az_container and az_blob:
                    data, ftype = _load_from_azure(az_container, az_blob, az_conn if az_conn else None)
                    if data is not None:
                        st.session_state['data'] = data
                        st.session_state['file_type'] = ftype
                        st.success(f"✅ Loaded from Azure: {len(data) if isinstance(data, pd.DataFrame) else 'text'}")
                else:
                    st.warning("Enter container and blob path.")
        uploaded_file = None
    else:
        data_url = st.text_input("Enter direct file URL (CSV/JSON/Parquet)")
        if st.button("Load from URL"):
            st.info("Direct URL loading is coming soon. Download the file and use Local File for now.")
        uploaded_file = None

    if 'data' not in st.session_state:
        st.session_state['data'] = None
        st.session_state['file_type'] = None

    if uploaded_file is not None:
        with st.spinner("Loading your data..."):
            data, ftype = load_data(uploaded_file)
            if isinstance(data, pd.DataFrame):
                st.session_state['data'] = data
                st.session_state['file_type'] = ftype
                st.success(f"✅ Loaded {len(data):,} rows × {len(data.columns):,} columns ({ftype})")

                # Enhanced info
                mem_kb = data.memory_usage(deep=True).sum() / 1024
                cols_num = len(data.select_dtypes(include=[np.number]).columns)
                cols_cat = len(data.select_dtypes(include=['object']).columns)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Rows", f"{len(data):,}")
                with c2: st.metric("Columns", f"{len(data.columns):,}")
                with c3: st.metric("Numeric Cols", f"{cols_num:,}")
                with c4: st.metric("Memory (KB)", f"{mem_kb:,.1f}")

                # Full data preview controls
                st.subheader("👀 Data Preview")
                show_full = st.checkbox("Show full dataset in preview (all rows & columns)", value=True)
                if show_full:
                    preview_df = data
                else:
                    r = st.slider("Rows", 5, min(1000, len(data)), min(100, len(data)))
                    c = st.slider("Columns", 5, len(data.columns), min(15, len(data.columns)))
                    preview_df = data.head(r).iloc[:, :c]
                st.dataframe(preview_df, use_container_width=True, height=min(600, max(300, len(preview_df) * 25)))

                # Note about streaming
                if streaming_enabled:
                    st.info("🔄 Streaming mode is enabled. For very large files, EMMA will process data incrementally.")
            else:
                # text/pdf fallback already handled in load_data
                st.session_state['data'] = data
                st.session_state['file_type'] = ftype

    # Chat interface
    st.markdown("### 💬 Chat with EMMA")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f'<div class="chat-bubble user-bubble">{chat["user"]}</div>', unsafe_allow_html=True)
        
        # Bot response
        bot_response = chat["bot"]
        
        # Handle inline plots first
        if "plot" in chat:
            # Generate a unique key for each plot based on chat index and timestamp
            plot_key = f"plot_{i}_{hash(str(chat.get('timestamp', datetime.now())))}"
            st.plotly_chart(chat["plot"], use_container_width=True, key=plot_key)
        
        # Then display text response
        if isinstance(bot_response, dict):
            # Display the text response if it exists
            if "response" in bot_response:
                st.markdown(f'<div class="chat-bubble bot-bubble">{bot_response["response"]}</div>', unsafe_allow_html=True)
            
            # Handle tabular data
            elif "tabular_data" in bot_response:
                st.dataframe(bot_response["tabular_data"], use_container_width=True)
        else:
            # Display string response
            st.markdown(f'<div class="chat-bubble bot-bubble">{bot_response}</div>', unsafe_allow_html=True)
    
    # Chat input
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    # Create a unique key for the input field
    input_key = f"user_input_{st.session_state.input_key}"
    
    # Add JavaScript for Enter key support
    st.markdown("""
    <script>
    // Function to handle Enter key press
    function handleEnterKey() {
        const textInputs = document.querySelectorAll('input[data-testid="stTextInput"]');
        textInputs.forEach(input => {
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    // Find the submit button and click it
                    const submitButton = document.querySelector('button[data-testid="baseButton-secondary"]');
                    if (submitButton) {
                        submitButton.click();
                    }
                }
            });
        });
    }
    
    // Run the function when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', handleEnterKey);
    } else {
        handleEnterKey();
    }
    
    // Also run on Streamlit rerun
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                handleEnterKey();
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Use text_area for better multi-line support and Enter key handling
    with st.form(key=f"chat_form_{st.session_state.input_key}"):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_area(
                "Ask EMMA anything about your data:", 
                key=input_key, 
                placeholder="e.g., 'Show me people under 30' or 'What's the average salary?' (Press Shift+Enter for new line, Enter to send)",
                height=100,
                max_chars=2000
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("➤", use_container_width=True)
    
    # Handle input submission
    if submit_button and user_input:
        # Check if this is a duplicate of the last question
        if (st.session_state.chat_history and 
            st.session_state.chat_history[-1]["user"] == user_input):
            st.warning("You just asked this question. Please ask something different or wait a moment.")
        else:
            with st.spinner("🤔 EMMA is thinking..."):
                data = st.session_state.get('data') if st.session_state.get('data') is not None and isinstance(st.session_state['data'], pd.DataFrame) else None
                
                if data is not None:
                    # Check if this is a visualization request
                    viz_keywords = [
                        "plot", "chart", "graph", "visualize", "show me", "create", "generate", "draw", "display", "make",
                        "histogram", "box plot", "violin plot", "kde plot", "distribution", "density", "spread",
                        "bar chart", "count plot", "pie chart", "donut chart", "strip plot", "frequency", "proportion",
                        "scatter plot", "bubble chart", "line chart", "pair plot", "correlation", "relationship", "trend",
                        "heatmap", "correlation matrix", "clustermap", "pivot table",
                        "time series", "trend", "area chart", "progression", "seasonal", "temporal",
                        "qq plot", "residual plot", "ecdf plot", "skewness", "kurtosis",
                        "side by side", "comparison", "grouped", "aggregated", "buckets", "ranges", "segments",
                        "profiling", "analysis", "exploration", "investigation", "examination", "study"
                    ]
                    
                    # Enhanced visualization detection - include heatmap and more specific terms
                    explicit_viz_keywords = [
                        "pie chart", "bar chart", "histogram", "scatter plot", "line chart", 
                        "heatmap", "heat map", "correlation matrix", "correlation heatmap",
                        "plot", "chart", "graph", "visualize", "show me a chart", "create a chart",
                        "generate a chart", "draw a chart", "display a chart", "make a chart",
                        "show the proportion", "show the distribution", "show the breakdown",
                        "show me", "create", "generate", "draw", "display", "make"
                    ]
                    
                    # Check if this is an explicit visualization request
                    is_viz_request = any(keyword in user_input.lower() for keyword in explicit_viz_keywords)
                    
                    if is_viz_request:
                        # Enhanced visualization prompt with explicit code generation
                        viz_prompt = f"""The user asked: "{user_input}"

🚨 CRITICAL INSTRUCTION: You MUST generate EXECUTABLE Python code that creates the actual visualization, NOT text descriptions or explanations.

❌ DO NOT DO THIS:
- "Here's the Python code using matplotlib..."
- "This chart will help us..."
- "To visualize this, I can generate..."
- "import matplotlib.pyplot as plt..."
- "Key insights from the heatmap:"
- "Would you like me to highlight..."

✅ DO THIS INSTEAD:
Generate ONLY a Python code block that creates the actual visualization.

REQUIRED FORMAT:
```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create the visualization
plt.figure(figsize=(12, 8))
# Your plotting code here
plt.title('Your Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
fig = plt.gcf()
```

CODE REQUIREMENTS:
- MUST use matplotlib.pyplot (plt) or seaborn (sns)
- MUST use 'data' as DataFrame variable name
- MUST end with 'fig = plt.gcf()'
- MUST be executable code that creates a real chart
- NO text explanations, NO insights, NO recommendations - ONLY code
- For heatmaps, use seaborn.heatmap() with proper formatting

EXAMPLE FOR HEATMAP:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Group data by City and Product, calculate average sales
city_product_sales = data.groupby(['City', 'Product'])['Total_Amount'].mean().unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(city_product_sales, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
plt.title("Average Sales by City and Product")
plt.xlabel("Product")
plt.ylabel("City")
plt.xticks(rotation=45)
plt.tight_layout()
fig = plt.gcf()
```

Generate ONLY the Python code block, nothing else."""

                        # Use default API (Groq) for visualization requests
                        groq_response = ask_groq(viz_prompt, model=st.session_state.model_choice, data=data, api_type="groq")
                        code = extract_code_from_response(groq_response)
                        
                        if code:
                            # Execute the visualization code with better error handling
                            local_vars = {
                                'data': data, 
                                'pd': pd, 
                                'np': np, 
                                'plt': __import__('matplotlib.pyplot'),
                                'sns': __import__('seaborn')
                            }
                            
                            fig = None
                            error_msg = None
                            
                            # Execute matplotlib code with comprehensive error handling
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                try:
                                    # Add import statement if missing
                                    if 'import matplotlib.pyplot as plt' not in code:
                                        code = 'import matplotlib.pyplot as plt\n' + code
                                    
                                    exec(code, {}, local_vars)
                                    
                                    if 'fig' in local_vars:
                                        matplotlib_fig = local_vars['fig']
                                        # Convert matplotlib figure to plotly for display
                                        import plotly.tools as tls
                                        plotly_fig = tls.mpl_to_plotly(matplotlib_fig)
                                        fig = plotly_fig
                                    else:
                                        # Try to get the current figure
                                        import matplotlib.pyplot as plt
                                        if plt.get_fignums():
                                            matplotlib_fig = plt.gcf()
                                            import plotly.tools as tls
                                            plotly_fig = tls.mpl_to_plotly(matplotlib_fig)
                                            fig = plotly_fig
                                        else:
                                            error_msg = "No figure was created"
                                            
                                except Exception as e:
                                    error_msg = f"Visualization failed: {str(e)}"
                                    # Try to create a simple fallback visualization
                                    try:
                                        import matplotlib.pyplot as plt
                                        plt.figure(figsize=(10, 6))
                                        if 'salary' in data.columns:
                                            plt.hist(data['salary'], bins=10, alpha=0.7, color='skyblue')
                                            plt.title('Salary Distribution (Fallback)')
                                            plt.xlabel('Salary')
                                            plt.ylabel('Frequency')
                                        elif len(data.select_dtypes(include=[np.number]).columns) > 0:
                                            numeric_col = data.select_dtypes(include=[np.number]).columns[0]
                                            plt.hist(data[numeric_col], bins=10, alpha=0.7, color='lightgreen')
                                            plt.title(f'{numeric_col} Distribution (Fallback)')
                                            plt.xlabel(numeric_col)
                                            plt.ylabel('Frequency')
                                        plt.tight_layout()
                                        matplotlib_fig = plt.gcf()
                                        import plotly.tools as tls
                                        plotly_fig = tls.mpl_to_plotly(matplotlib_fig)
                                        fig = plotly_fig
                                        error_msg = f"Original visualization failed: {str(e)}. Showing fallback chart."
                                    except Exception as fallback_error:
                                        error_msg = f"Both original and fallback visualizations failed: {str(e)} | {str(fallback_error)}"
                            
                            if fig is not None:
                                # Extract text analysis (remove code blocks)
                                text_response = groq_response
                                if '```' in text_response:
                                    parts = text_response.split('```')
                                    text_response = parts[0] + parts[-1] if len(parts) > 2 else parts[0]
                                
                                st.session_state.chat_history.append({
                                    "user": user_input, 
                                    "bot": text_response.strip(), 
                                    "plot": fig, 
                                    "timestamp": datetime.now()
                                })
                            else:
                                st.session_state.chat_history.append({
                                    "user": user_input, 
                                    "bot": f"{groq_response}\n\n❌ {error_msg}", 
                                    "timestamp": datetime.now()
                                })
                        else:
                            st.session_state.chat_history.append({
                                "user": user_input, 
                                "bot": "❌ No visualization code found in response.", 
                                "timestamp": datetime.now()
                            })
                    else:
                        # Regular analysis request - use conversational approach
                        groq_response = ask_groq(user_input, model=st.session_state.model_choice, data=data, api_type="groq")
                        st.session_state.chat_history.append({"user": user_input, "bot": groq_response})
                    
                    # Save to database
                    if 'current_session_id' in st.session_state:
                        save_chat_message(st.session_state.current_session_id, user_input, str(st.session_state.chat_history[-1]["bot"]))
                else:
                    # No data uploaded - use conversational approach
                    groq_response = ask_groq(user_input, model=st.session_state.model_choice, api_type="groq")
                    st.session_state.chat_history.append({"user": user_input, "bot": groq_response})
                    if 'current_session_id' in st.session_state:
                        save_chat_message(st.session_state.current_session_id, user_input, str(groq_response))
                
                st.session_state.input_key += 1
                st.rerun()

if __name__ == "__main__":
    main()