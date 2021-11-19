import gradio as gr
from spacy import displacy

from transformers import AutoTokenizer, AutoModelForTokenClassification,pipeline
tokenizer = AutoTokenizer.from_pretrained("abhibisht89/spanbert-large-cased-finetuned-ade_corpus_v2")
model = AutoModelForTokenClassification.from_pretrained("abhibisht89/spanbert-large-cased-finetuned-ade_corpus_v2").to('cpu')
adr_ner_model = pipeline(task="ner", model=model, tokenizer=tokenizer,grouped_entities=True)      

def get_adr_from_text(sentence):
    tokens = adr_ner_model(sentence)
    entities = []
    
    for token in tokens:
        label = token["entity_group"]
        if label != "O":
            token["label"] = label
            entities.append(token)
    
    params = [{"text": sentence,
               "ents": entities,
               "title": None}]
    
    html = displacy.render(params, style="ent", manual=True, options={
        "colors": {
                   "DRUG": "#f08080",
                   "ADR": "#9bddff",
               },
    })
    return html

exp=["Abortion, miscarriage or uterine hemorrhage associated with misoprostol (Cytotec), a labor-inducing drug.",
    "Addiction to many sedatives and analgesics, such as diazepam, morphine, etc.",
    "Birth defects associated with thalidomide",
    "Bleeding of the intestine associated with aspirin therapy",
    "Cardiovascular disease associated with COX-2 inhibitors (i.e. Vioxx)",
    "Deafness and kidney failure associated with gentamicin (an antibiotic)",
    "Having fever after taking paracetamol"]

desc="An adverse drug reaction (ADR) can be defined as an appreciably harmful or unpleasant reaction resulting from an intervention related to the use of a medicinal product.\
 The goal of this project is to extracts the adverse drug reaction from unstructured text with the Drug."

inp=gr.inputs.Textbox(lines=5, placeholder=None, default="", label="text to extract adverse drug reaction and drug mention")
out=gr.outputs.HTML(label=None)

iface = gr.Interface(fn=get_adr_from_text, inputs=inp, outputs=out,examples=exp,article=desc,title="Adverse Drug Reaction Xtractor",theme="huggingface",layout='horizontal')
iface.launch()