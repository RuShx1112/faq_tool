# FAQ Answer Tool

### How it works


1. User question

2. Embedding  (sentence-transformers)

3. Cosine similarity search 

4. Top-3 FAQ chunks retrieved 

5. API call with context 

6. Answer + sources

## Setup

### 1. Clone and enter the repo

git clone https://github.com/RuShx1112/faq_tool.git

cd faq_tool


### 2. Create and activate a virtual environment


- python -m venv .venv

- .venv\Scripts\activate        


### 3. Install dependencies and Run


pip install groq

set GROQ_API_KEY= groq api key

python api.py



### 4. How it answers:
You: is IVF painful?

Answer:
IVF egg retrieval is done under sedation, so it's pain-free during the procedure.
The hormone injections are subcutaneous and brief — most women find them manageable.


## Assumptions

- No conversation history is maintained. Each question is independent.
- Questions that are asked haev somewhat relevance to faqs from given dataset


## Limitations and what I'd improve with more time

- No conversation history
- Memory doesnt exist, cant ask a follow-up Question using 'it'(becasue no previous context)
- No answers to generic questions apart from what can be obtained from the faq doc(eg: what is ivf?)
