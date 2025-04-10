
from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
SECRET_KEY = 'supersecretkey'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

embedding_model = OpenAIEmbeddings()
vectorstore = None
qa_chain = None

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a proposal writer generating responses to a government RFP.
Use the context below to answer the question as a draft proposal section.

Context:
{context}

Question:
{question}

Draft Response:
"""
)

users = {'admin': 'password123'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    response = ""
    if request.method == 'POST':
        question = request.form['question']
        if qa_chain:
            response = qa_chain.run(question)
    return render_template('index.html', response=response)

@app.route('/upload', methods=['POST'])
def upload():
    global vectorstore, qa_chain
    if 'username' not in session:
        return redirect(url_for('login'))
    file = request.files['file']
    if not file or file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file')
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embedding_model)

    llm = OpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    flash('File uploaded and processed successfully!')
    return redirect(url_for('index'))

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
