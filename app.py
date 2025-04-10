
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
