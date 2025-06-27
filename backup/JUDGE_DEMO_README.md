# 🏆 Hackathon Judge Demo - Requirements Analysis Tool

## Quick Start for Judges

### One-Click Demo Launch
```bash
poetry run python run_judge_demo.py
```
This will:
- Start the backend API server  
- Launch the demo interface at http://localhost:8502
- Open your browser automatically
- Provide fallback mock mode if API fails

### Alternative Manual Launch
```bash
# Start demo interface only (uses mock results)
poetry run streamlit run judge_demo.py --server.port 8502
```

### Quick Setup Validation
```bash
# Verify Poetry is working
poetry --version

# Verify demo imports correctly
poetry run python -c "import judge_demo; print('✅ Demo ready!')"
```

## 🎯 What This Demo Shows

### **AI-Powered Requirements Analysis System**
- **Auto-selects diverse requirements** from 5 categories (Security, Performance, Functional, Integration, Budget)
- **Evidence-based analysis** using real documentation search
- **Multi-model AI evaluation** with IBM Granite 3 + Llama 3.3
- **Comprehensive scoring** with feasibility ratings and recommendations

### **Complete Analysis Pipeline**
1. **📝 Requirement Selection**: Smart auto-selection ensuring diversity across categories
2. **🔍 Evidence Search**: Vector database search through documentation 
3. **📊 AI Evaluation**: Multi-model analysis with cross-validation
4. **📋 Report Generation**: Detailed findings with actionable insights

## 🎬 Demo Flow

| Stage | What Judges See | Time |
|-------|----------------|------|
| **Setup** | Auto-selection of 3-5 diverse requirements | 10 seconds |
| **Analysis** | Real-time pipeline visualization | 45 seconds |
| **Results** | Comprehensive analysis report | Review time |

## 🔧 Technology Highlights

- **🧠 IBM watsonx AI**: Advanced language models (Granite 3-8B + Llama 3.3-70B)
- **📚 Vector Database**: ChromaDB with semantic search capabilities  
- **🎯 Reranking**: Cross-encoder models for evidence relevance scoring
- **🌐 Modern Interface**: Streamlit frontend + FastAPI backend
- **📊 Evidence-Based**: All recommendations backed by document analysis

## 📋 Sample Requirements Categories

### 🔒 **Security**
- Multi-factor authentication with TOTP tokens
- SSL/TLS encryption with certificate validation
- AES-256 encryption with key rotation

### ⚡ **Performance** 
- Handle 1000+ concurrent users sub-2-second response
- Auto-scaling based on CPU/memory utilization
- 99.9% uptime with disaster recovery

### 🛠️ **Functional**
- Drag-and-drop report designer
- Real-time dashboard updates
- REST API integrations

### 🔗 **Integration**
- Apache Kafka event streaming
- Webhook notifications
- Audit trail logging

### 💰 **Budget**
- $500K implementation constraint
- Cloud cost optimization

## 🎯 Judge Evaluation Points

### **Innovation**
- Evidence-based AI analysis (not just rule-based)
- Multi-model cross-validation approach
- Automated requirement categorization

### **Technical Excellence**  
- Vector database integration
- Real-time reranking algorithms
- Modern microservices architecture

### **Business Value**
- Reduces requirement analysis time from days to minutes
- Evidence-backed recommendations increase confidence
- Scalable to enterprise documentation sets

### **Demo Quality**
- Auto-selects diverse, realistic requirements
- Shows complete end-to-end pipeline
- Provides both live and mock modes for reliability

## 🚀 Quick Demo Script for Judges

1. **Start**: Click "🚀 Start Demo" 
2. **Select**: Choose 3-5 requirements (recommendation: 4)
3. **Review**: See auto-selected diverse requirements across categories
4. **Analyze**: Click "🔍 Begin AI Analysis" to see pipeline
5. **Results**: Review comprehensive analysis report with feasibility scores

**Total Demo Time**: ~2-3 minutes including explanation

## ⚡ Fallback Mode

If the API server fails to start, the demo automatically provides:
- **Mock analysis results** showing expected output format
- **Pipeline visualization** demonstrating the analysis flow  
- **Complete UI experience** without requiring backend dependencies

This ensures judges can see the full demo even in constrained environments.

## 🔧 Technical Requirements

- **Python 3.11+**
- **Poetry 2.1+** (for dependency management)
- **Dependencies**: Installed via `poetry install`
- **Optional**: IBM watsonx API credentials (demo works without them in mock mode)
- **Browser**: Any modern browser for the web interface

## 📦 Setup for Judges

If running for the first time:
```bash
# Install dependencies
poetry install

# Launch demo
poetry run python run_judge_demo.py
```

---

**🏆 Ready for judging!** The demo showcases a production-ready AI system for requirements analysis with real business applications. 