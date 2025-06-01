from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def merge_subwords(entities):
    merged = []
    temp = {"word": "", "entity_group": "", "score": 0.0, "count": 0}

    for ent in entities:
        word = ent["word"]
        if word.startswith("##"):
            temp["word"] += word[2:]
            temp["score"] += ent["score"]
            temp["count"] += 1
        else:
            if temp["word"]:
                # Finalize previous word
                temp["score"] /= temp["count"]
                merged.append({
                    "word": temp["word"],
                    "entity_group": temp["entity_group"],
                    "score": round(temp["score"], 3)
                })
            temp = {
                "word": word,
                "entity_group": ent["entity_group"],
                "score": ent["score"],
                "count": 1
            }

    if temp["word"]:
        temp["score"] /= temp["count"]
        merged.append({
            "word": temp["word"],
            "entity_group": temp["entity_group"],
            "score": round(temp["score"], 3)
        })

    return merged

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

    # Convert numpy.float32 to float
    for ent in raw_result:
        ent["score"] = float(ent["score"])

    # ✅ Merge subwords
    return {"entities": merge_subwords(raw_result)}

@app.get("/")
def read_root():
    return {"message": "NER API is live!"}
