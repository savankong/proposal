
import logging
from flask import Flask, request, render_template, redirect, url_for, session, flash
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os

logging.basicConfig(level=logging.DEBUG)

SECRET_KEY = 'supersecretkey'

app = Flask(__name__)
app.secret_key = SECRET_KEY

embedding_model = OpenAIEmbeddings()
vectorstore = None
qa_chain = None

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a proposal writer generating responses to a government RFP.
Use the context below to answer the question as a draft proposal section.

Context:
{context}

Question:
{question}

Draft Response:
"""
)

users = {'admin': 'password123'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    response = ""
    if request.method == 'POST':
        if 'rfp_text' in request.form:
            try:
                rfp_text = request.form['rfp_text']
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents([Document(page_content=rfp_text)])
                global vectorstore, qa_chain
                vectorstore = FAISS.from_documents(docs, embedding_model)
                llm = OpenAI(temperature=0.2)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt_template}
                )
                flash("RFP text submitted and processed!")
            except Exception as e:
                logging.exception("Failed to process RFP text")
                flash(f"Error processing RFP: {e}")
        elif 'question' in request.form:
            try:
                question = request.form['question']
                if qa_chain:
                    response = qa_chain.run(question)
            except Exception as e:
                logging.exception("Error generating response")
                flash(f"Error: {e}")
    return render_template('index.html', response=response)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
