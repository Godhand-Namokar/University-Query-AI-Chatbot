# 🎓 University Query AI Chatbot

## 🚀 Overview
This project is an AI-powered role-based chatbot designed to handle university-related queries efficiently.

It provides personalized responses to students, parents, faculty, and admin users using AI and structured academic data.

---

## 🤖 Features
- 🔐 Role-based authentication (Student / Parent / Faculty / Admin)
- 📱 Telegram Bot integration
- 🧠 AI-powered question answering using LangChain & HuggingFace
- 📅 Timetable and holiday query system
- ⚡ Fast semantic search using FAISS
- 📊 Real-time data handling using Excel

---

## 🛠️ Tech Stack
- Python
- LangChain
- HuggingFace Transformers
- FAISS
- Telegram Bot API
- Pandas

---
## 📂 Project Structure
University-Query-AI-Chatbot/
│
├── mainpro.py              # Main chatbot application
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── poster.png             # Project poster
├── README.md              # Project documentation
│
├── Databaseai.xlsx.xlsx   # User database (roles & phone numbers)
├── BTech_Sem2_Timetable.xlsx   # Timetable data
├── MITVPU_Holidays_Important_Dates.xlsx  # Holidays data
│
└── screenshots/           # Telegram bot screenshots (optional)
    ├── bot1.png
    └── bot2.png

## 📂 Project Files
- `mainpro.py` → Main chatbot logic  
- `.env.example` → Environment variables setup  
- `requirements.txt` → Dependencies  
- `poster.png` → Project poster  

---

## 📌 Project Poster
![Poster](poster.png)

---

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python mainpro.py
