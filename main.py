import os

# pip install langchain
# pip install langchain.chains
# pip install langchain_huggingface
# pip install huggingface_hub

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# enter huggingface api key here
HUGGINGFACEHUB_API_TOKEN = "xxxxxxxxxx"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

question = "describe the child hood of winston churchill until the age of 18?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
llm_chain = prompt | llm
print(llm_chain.invoke({"question": question}))
