# SHL Research Intern - Recommender System Optimization

## Project Overview
This repository contains my submission for the **SHL Research Intern Assessment**, where I developed and optimized a **recommender system**.  
The main objective of this project was to build a high-performing model capable of accurately recommending items based on user–item interactions and to improve its overall performance score through systematic experimentation and tuning.

The project includes:
- My complete implementation notebook.  
- A 2-page report explaining my approach and optimization process.  
- Supporting files for data preprocessing and evaluation.

---

## My Approach
I began with a simple baseline recommender model and gradually enhanced it through several optimization stages. My workflow was structured into four main phases:

1. **Data Cleaning & Preprocessing**  
   - I handled missing values, normalized numerical columns, and ensured consistency in user–item interaction data.  
   - I also transformed categorical features into numerical representations suitable for modeling.

2. **Model Development**  
   - I implemented a baseline collaborative filtering model.  
   - Then, I extended it into a hybrid model by integrating content-based features with user-item embeddings.

3. **Optimization Phase**  
   - I experimented with different algorithms, including Matrix Factorization and LightFM.  
   - Using grid search and parameter tuning, I optimized key parameters like learning rate, embedding size, and regularization strength.  
   - I continuously evaluated the model using precision@k, recall@k, and F1-score metrics.

4. **Performance Improvement**  
   - I analyzed the weaknesses of my initial results and refined the model accordingly.  
   - Through iterative fine-tuning, I managed to achieve a significant improvement in the overall performance score.

---

## Results

📊 MODEL EVALUATION RESULTS
-----------------------------
- Users Evaluated: 6
- Average Precision: 0.8649
- Average Recall: 0.963
- Average F1: 0.8396

---

## Key Learnings
Working on this project taught me how to balance **model accuracy** and **computational efficiency**.  
I also realized the importance of proper data preprocessing and parameter tuning in achieving better real-world performance.  
This experience helped me understand how research-driven optimization can enhance even simple recommendation systems.

---

## Repository Structure

├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebook with full implementation
├── report/ # Final 2-page optimization report (DOCX)
├── src/ # Core scripts for data prep and model training
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Tech Stack
- **Language:** Python,HTML,CSS
- **Libraries:** flask,pandas,scikit-learn,openpyxl,numpy,faiss-cpu,openai,google-generativeai,langchain-groq
- **Environment:** vscode 

---

## Author
**Yuvan Gowtham**  
Research AI Intern Candidate – SHL  
 Email: [yuvangoutham12@gmail.com]  
 Location: India  

---
