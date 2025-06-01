from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

app = FastAPI()

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("ner_model")
tokenizer = AutoTokenizer.from_pretrained("ner_model")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    raw_result = nlp(data.text)

    # Fix: Convert numpy.float32 to Python float
    for ent in raw_result:
        ent["score"] = float(ent["score"])

    return {"entities": raw_result}

@app.get("/")
def read_root():
    return {"message": "NER API is live!"}
