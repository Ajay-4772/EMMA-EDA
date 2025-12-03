# 🤖 EMMA: Enhanced EDA Chatbot with Advanced Features

## Overview
EMMA (Enhanced Machine Learning Assistant) is a powerful web application designed to facilitate Exploratory Data Analysis (EDA) through natural language queries. Built with Streamlit and advanced data processing capabilities, EMMA provides an intuitive, feature-rich interface for data exploration and visualization.

## ✨ Key Features

### 🎨 **Theme Support**
- **Dark & Light Mode Toggle** - Switch between themes with one click
- **Appwrite-inspired UI Design** - Modern, clean interface
- **Pure white text in dark mode** - Optimal contrast and readability
- **Adaptive font sizing** - Responsive design that adapts to system preferences

### 📊 **Advanced Visualizations**
- **Multiple Chart Types**: Bar Charts, Pie Charts, Line Charts, Box Plots, Scatter Plots, Histograms
- **Interactive Charts** - Powered by Plotly for rich interactions
- **Download Options** - Save charts as PNG or PDF with dedicated buttons
- **Fullscreen View** - Enhanced viewing experience for detailed analysis

### 📁 **Multiple File Format Support**
- **CSV, Excel (.xlsx, .xls)**
- **JSON, TSV, Parquet**
- **PDF, TXT** (with text extraction)
- **Auto-detection** and processing with pandas

### 💾 **Downloadable Outputs**
- **Cleaned datasets** - Export filtered results as CSV
- **Chart downloads** - Save visualizations as PNG/PDF
- **Session summaries** - Download complete chat history as text
- **Data exports** - Export processed datasets

### 🗃️ **Chat History System**
- **SQLite Database** - Persistent storage of all conversations
- **Session Management** - Organize chats by dataset and timestamp
- **Auto-generated titles** - Smart session naming
- **Message persistence** - Never lose your analysis

### 🔍 **Searchable Chat History**
- **Keyword search** - Find specific conversations
- **Date/time filtering** - Browse by time period
- **File-based search** - Find chats by dataset name
- **Fuzzy matching** - Intelligent search algorithms

### 📋 **SQL-Like Tabular Output**
- **Structured data display** - Clean, organized table views
- **Filtered results** - Show specific data subsets
- **Sortable columns** - Interactive data exploration
- **Export capabilities** - Download filtered results

### ⚡ **Quick EDA Actions**
- **One-click analysis** - Summary stats, missing values, correlations
- **Outlier detection** - Automatic identification of data anomalies
- **Data health scoring** - Quality assessment with actionable insights
- **Smart suggestions** - Context-aware recommendations

## 🚀 Installation

```bash
git clone <repository-url>
cd EDA-Chatbot-using-LangChain-Streamlit-and-LLMs-for-Natural-Language-Data-Exploration
pip install -r requirements.txt
```

## 🎯 Usage

```bash
streamlit run src/app.py
```

## 📁 Project Structure
```
EMMA-CONVERSATIONAL-EDA

EMMA-Converesional-EDA-Assistant
├── src/
│   ├── app.py                 # Main Streamlit application
│   └── llm/
│       └── llm_api.py         # Generic LLM integration (supports Ollama)
├── requirements.txt       # Python dependencies
├── README.md                 # Project documentation
├── .streamlit/               # Streamlit configuration
│   └── config.toml
└── chat_history.db         # SQLite database for chat persistence
```

## 💬 Example Queries

### Data Analysis
- "Show me people under 30"
- "What's the average salary?"
- "List all people with salary > 50000"
- "Give me a summary of the dataset"

### Visualizations
- "Visualize the salary distribution"
- "Create a bar chart of ages"
- "Show me a scatter plot of age vs salary"
- "Generate a pie chart of categories"

### Advanced Queries
- "Find outliers in the data"
- "Show correlation matrix"
- "Check for missing values"
- "Export filtered results"

## 🎨 Theme Features

### Light Mode
- Clean white background
- High contrast text
- Professional appearance
- Optimized for daytime use

### Dark Mode
- Pure white text on dark background
- Reduced eye strain
- Modern aesthetic
- Perfect for low-light environments

## 📊 Visualization Gallery

EMMA supports a wide range of chart types:
- **Bar Charts** - For categorical data comparison
- **Pie Charts** - For proportion visualization
- **Line Charts** - For trend analysis
- **Scatter Plots** - For correlation exploration
- **Box Plots** - For distribution analysis
- **Histograms** - For frequency distribution
- **Correlation Matrices** - For relationship analysis

## 🔧 Technical Features

- **Responsive Design** - Works on desktop and mobile
- **Real-time Processing** - Instant analysis and visualization
- **Memory Efficient** - Optimized for large datasets
- **Error Handling** - Graceful error recovery
- **Session Persistence** - Maintains state across browser sessions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

**Made with ❤️ by EMMA - Your Intelligent Data Analysis Assistant**