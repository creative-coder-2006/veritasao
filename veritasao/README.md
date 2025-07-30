# 🤖 VERITAS.AI: The AI-Powered Truth Toolkit

**VERITAS.AI** is a comprehensive, production-ready application designed to combat digital misinformation. It leverages a sophisticated ensemble of AI and machine learning models to analyze content from multiple platforms (news articles, Reddit posts, and YouTube videos), providing users with a clear and actionable assessment of its trustworthiness and origin.

The system is built with resilience in mind, featuring intelligent fallback mechanisms that ensure core functionality is always available, even without external API keys.

---

## 🚀 Key Features

### 🔍 Multi-Platform Analysis
* **News Articles:** Real-time scraping with `newspaper3k` and a robust `requests`/`BeautifulSoup` fallback.
* **Reddit Posts:** Official API integration via `PRAW` with a seamless simulation fallback for offline use.
* **YouTube Videos:** Deep content extraction using `yt-dlp`, enabling separate analysis of audio and video streams.

### 🧠 Advanced AI Detection
* **Misinformation Detection:** Fine-tuned BERT-based models with confidence scoring to assess factual accuracy.
* **LLM Origin Detection:** A powerful ensemble approach combining statistical analysis and a dedicated AI text detection model.
* **Deepfake Detection:** Advanced computer vision analysis of video frames using a pre-trained `R3D_18` model.
* **Audio Analysis:** High-accuracy `Whisper` transcription coupled with anomaly detection to identify audio manipulation.

### 📊 Comprehensive Scoring System
* **Misinformation Score:** An ML-based probability of how likely the content is to be misinformation.
* **Confidence Score:** An indicator of the AI models' certainty in their own predictions.
* **Trust Score:** A holistic, multi-factor assessment of the content's overall trustworthiness.
* **Credibility Score:** An evaluation of the source and content quality.
* **LLM Origin Probability:** A clear score indicating the likelihood of AI generation.

### 🚩 Platform-Specific Flagging
* **News Sources:** Users can flag unreliable domains, contributing to a community-driven reputation system.
* **Subreddits & YouTube Channels:** The system architecture supports future extensions for community and creator-level risk tracking.

### 🧠 OpenAI Explainable AI (XAI)
* **Expert Prompts:** Utilizes platform-specific master prompts to guide the LLM into the role of a domain expert (e.g., "news analyst," "digital forensics expert").
* **Detailed Explanations:** Translates complex numerical scores into clear, human-readable summaries, making the analysis accessible to everyone.

### 🔐 Security & Privacy
* **User Authentication:** Secure user login and registration with password hashing via `bcrypt`.
* **Local Database:** Uses `SQLite` for a private, self-contained database, ensuring user data remains local.
* **Temporary File Cleanup:** Automatic management and cleanup of downloaded media files.

---

## 🛠️ Tech Stack

* **Main Application:** Streamlit
* **AI & ML:** PyTorch, Transformers, Scikit-learn, Whisper, Librosa
* **Scraping:** Newspaper3k, PRAW, yt-dlp, BeautifulSoup4
* **Database:** SQLite
* **Security:** Bcrypt
* **Other:** OpenAI, NumPy, OpenCV

---

## 📂 File Structure