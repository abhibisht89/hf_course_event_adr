from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import pickle
import pandas as pd
import gradio as gr

bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
corpus_embeddings=pd.read_pickle("corpus_embeddings_cpu.pkl")
corpus=pd.read_pickle("corpus.pkl")

def search(query,top_k=100):
    print("Top 5 Answer by the NSE:")
    print()
    ans=[]
    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, corpus[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    
    for idx, hit in enumerate(hits[0:5]):
        ans.append(corpus[hit['corpus_id']])
    return ans[0],ans[1],ans[2],ans[3],ans[4]

exp=["Who is steve jobs?","What is coldplay?","What is a turing test?","What is the most interesting thing about our universe?","What are the most beautiful places on earth?"]

desc="This is a semantic search engine powered by SentenceTransformers (Nils_Reimers) with a retrieval and reranking system on Wikipedia corpus. This will return the top 5 results. So Quest on with Transformers."

inp=gr.inputs.Textbox(lines=1, placeholder=None, default="", label="search you query here")
out=gr.outputs.Textbox(type="auto",label="search results")

iface = gr.Interface(fn=search, inputs=inp, outputs=[out,out,out,out,out],examples=exp,article=desc,title="Neural Search Engine",theme="huggingface",layout='vertical')
iface.launch(share=True)