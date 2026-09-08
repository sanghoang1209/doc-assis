# 🏢 Enterprise Knowledge Assistant

## 📖 The Story & Business Challenge

In almost every organization, vital knowledge is trapped inside hundreds of unorganized files — standard operating procedures (SOPs), project reports, market research, technical guides, and legal contracts.

This creates severe operational bottlenecks:
* **Wasted Productivity:** Employees spend **up to 30% of their workday** searching for specific answers across scattered documents.
* **Delayed Decision-Making:** Key stakeholders wait hours or days for team members to locate and verify historical records.
* **Knowledge Silos:** Critical operational knowledge stays isolated with key individuals instead of being accessible across the organization.

**This project was created to solve that fundamental business problem.**

---

## 💡 The Solution: An Autonomous Knowledge Partner

Imagine having a dedicated, tireless **Knowledge Assistant** sitting next to your team:

1. **Reasoning Before Answering:** When asked a question, the Assistant doesn't guess or hallucinate. It autonomously scans the document registry, selects relevant files, reads matching paragraphs, and synthesizes a clear, accurate answer based strictly on verified facts.
2. **Total Transparency & Trust:** Every answer includes a step-by-step **Reasoning Trace**. You can see exactly which documents were scanned, which sections were read, and why the Assistant reached its conclusion.
3. **Instant On-Demand Insights:** Empowers leadership and team members to extract actionable answers from internal documentation in seconds, not hours.

---

## 🎯 Business Value

* ⚡ **Accelerate Operations:** Reduce information search time from hours to seconds.
* 🛡️ **Zero Hallucination Risk:** Answers are grounded strictly in your organization's internal document context.
* 🔍 **Auditable & Transparent:** Full visibility into the Assistant's reasoning steps and source citations.
* 🔒 **Data Privacy First:** Keeps internal business knowledge private within your dedicated database environment.

---

## 🚀 Quick Start Guide

### 1. Launch Database
Start the local database service with vector capabilities:
```bash
docker compose up -d db
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/docqa
OLLAMA_BASE_URL=http://localhost:11434
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start the Application
Install dependencies and run the server:
```bash
uv sync
uv run uvicorn app.main:app --reload
```

### 4. Experience the Assistant
Open your browser and navigate to:
```text
http://localhost:8000/ui
```
Upload your business documents (SOPs, guides, reports) and start asking questions!
