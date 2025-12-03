<p align="center">🚀 EMMA EDA – A Conversational Exploratory Data Analysis Assistant</p>
<p align="center">Natural-language powered exploratory data analysis, visualizations, insights, and automated reasoning.</p>
<p align="center"> <!-- Core Badges --> <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" /> <img src="https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit" /> <img src="https://img.shields.io/badge/LangChain-Framework-orange?logo=chainlink" /> <img src="https://img.shields.io/badge/Groq-LLaMA3-green?logo=groq" /> <img src="https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite" /> <img src="https://img.shields.io/badge/License-MIT-yellow" /> <!-- Stylish Badges -->
<br><br>
<img src="https://img.shields.io/badge/EMMA-EDA%20Assistant-0A84FF?style=for-the-badge" />
<img src="https://img.shields.io/badge/Powered%20by-AI-black?style=for-the-badge" />
<img src="https://img.shields.io/badge/Conversational-EDA-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Made%20with%20❤️-by%20Ajay-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Data%20Analysis-Automated-success?style=for-the-badge" />

</p>

🚀 Overview

EMMA (Exploratory Machine-learning Model Assistant) is a conversational EDA system that allows users to analyze datasets using natural language instead of writing code. EMMA integrates:

Streamlit for the UI

LangChain for prompt routing

Groq LLaMA-3 / GPT-4 API for reasoning

Pandas + Plotly for data processing

SQLite for session history

EMMA transforms EDA from manual coding → interactive, AI-driven conversation.

It automates:

✔ Statistical summaries
✔ Data cleaning & preprocessing
✔ Correlation analysis
✔ Outlier detection
✔ Visualizations (bar, pie, scatter, heatmap, box plot)
✔ Full conversational explanations
✔ Persistent chat history

🧠 Key Capabilities

🗂️ 1. Upload Any Dataset

Supports:

CSV, TSV

Excel (.xls, .xlsx)

JSON

Parquet

TXT & PDF (extraction)

Auto-detects schema & infers data types.

💬 2. Conversational Analysis

Ask questions like:

“Show the top 10 customers by amount spent.”
“Plot a pie chart of product categories.”
“Find outliers in the salary column.”
“Display correlation heatmap.”

EMMA → interprets → generates code → visualizes → explains.

📊 3. Visualization Engine

Powered by Plotly:

Bar charts

Pie charts

Heatmaps

Line charts

Scatter plots

Boxplots

Supports:

✨ Full-screen
✨ Download (PNG/PDF)
✨ Hover interactions

🔍 4. Quick EDA Tools

Missing value detection

Outlier detection

Statistical summaries

Correlation matrices

Data quality insights

🗄️ 5. Chat History & Session Management

Conversation saved in SQLite

Auto session creation

Searchable history

Clean session grouping

🎨 6. Modern UI

Light & Dark mode

Clean sidebar layout

Inspired by Appwrite & Vercel

Responsive fonts


📦 Installation

1️⃣ Clone the repo
git clone https://github.com/<your-username>/EMMA-EDA.git
cd EMMA-EDA

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Create environment file
cp .env.example .env


Add your Groq / OpenAI keys inside .env

▶️ Run the App
streamlit run src/app.py


App URL:
👉 http://localhost:8501/

🧪 Example EDA Queries
🔹 Basic Queries

“Show summary statistics”

“Display the first 10 rows”

“Find missing values”

🔹 Visualization

“Plot salary distribution histogram”

“Scatter plot age vs salary”

“Create a pie chart of categories”

🔹 Intermediate

“Find outliers in the sales column”

“Show correlation heatmap”

“Plot monthly revenue trend”

🔹 Advanced

“Generate a box plot grouped by region”

“Compare warehouse vs retail sales”

“Analyze customer spending patterns”

⚙️ Tech Stack
🖥️ Frontend

Streamlit

Plotly

🧩 Backend

Python

Pandas, NumPy

LangChain

🤖 LLM Layer

Groq LLaMA-3

GPT-4 (optional)

🗄 Database

SQLite for chat history

🔐 Security

.env excluded from GitHub

API keys stored safely

SQLite DB can be reset anytime

EMMA does not upload user data externally

📄 License

This project is licensed under the MIT License.

❤️ Developed by

Ajay M

Team Members

Muralidharan R
Krishna K

