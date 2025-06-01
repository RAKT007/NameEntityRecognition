from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

app = FastAPI()

# Load model
model = AutoModelForTokenClassification.from_pretrained("ner_model")
tokenizer = AutoTokenizer.from_pretrained("ner_model")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    result = nlp(data.text)
    return {"entities": result}

@app.get("/")
def read_root():
    return {"message": "NER API is live!"}
