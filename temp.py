from utils import docsearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import os
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai
import sys
from utils import clean_text
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    query = "Where is America Loacted"
    query = "Explain testLesson function .Only give the code between `` and nothing else"
    query = "Analyze the function getTitle() in the call AuthBypass fot vulnerabilities"
    query = "What is getDefaultCategory"

    prompt_template = """Use the following pieces of context to answer the question at the end.Also answer the question in brief. If the answer is not from the context, just say "no", don't try to make up an answer.
    {context}
    Question: {question}
    Answer in English:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
        )
    retriever=docsearch.as_retriever()
    docs = retriever.get_relevant_documents(query)
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff", prompt=PROMPT)
    ans = chain.run(input_documents=docs, question=query)
    print(ans)

    if temp == "no":
        print("You need to continue")
    else:
        print(ans)
        sys.exit(0)

