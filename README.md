# 🧠 Doogie AI – Multi-Language Medical Reasoning API  

### A FastAPI-based backend for multilingual medical understanding powered by NHS & Oxford clinical data  

---

## 🚀 Overview  

**Doogie AI** is an intelligent backend API that allows users (patients or clinicians) to chat in **any language** about medical symptoms or health conditions.  
The system automatically:  
- Detects the user’s input language  
- Translates it to English  
- Uses **Oxford** and **NHS** clinical knowledge (stored locally)  
- Generates a structured medical reasoning response using the **Doogie Master Prompt**  
- Translates the response back to the user’s original language  

This backend is designed to integrate easily with a modern web frontend (to be provided by the client).

---

## 🧩 Core Features  

- 🌍 **Multi-Language Understanding** – detect and respond in any language  
- 🏥 **NHS + Oxford Reference Data** – local medical knowledge base  
- 🧠 **Doogie Master Prompt Integration** – structured reasoning for clinical-style responses  
- ⚡ **FastAPI Framework** – lightweight, scalable, deployable anywhere (Railway, Render, etc.)  
- 🔁 **Automatic Translation Flow** – English ⇄ User language  
- 🛡️ **Environment-Protected Keys** – OpenAI key handled via `.env`  


---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/MuhammadUzairMadni/DoogieAI_Final.git
cd DoogieAI_Final

