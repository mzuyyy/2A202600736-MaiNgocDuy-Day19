#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load biến môi trường
load_dotenv()

# Kết nối với Neo4j
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password
)

# Khởi tạo OpenAI client
openai_api_key = os.getenv("BTL_API_KEY")
openai_base_url = os.getenv("BTL_API_BASE_URL")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url
)

# Hàm để gọi API OpenAI
def call_openai_api(prompt: str) -> str:
    """Gọi API OpenAI để sinh văn bản."""
    try:
        response = client.chat.completions.create(
            model="btl-2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi gọi API: {e}"

# Tạo prompt template để truy vấn đồ thị
cypher_template = """
Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:

Schema: {schema}

Question: {question}
Cypher query:
"""

cypher_prompt = PromptTemplate.from_template(cypher_template)

# Tạo chain để sinh Cypher query
cypher_chain = (
    {
        "schema": lambda _: graph.get_schema,
        "question": RunnablePassthrough()
    }
    | cypher_prompt
    | RunnablePassthrough().assign(response=call_openai_api)
    | StrOutputParser()
)

# Hàm để truy vấn và trả lời câu hỏi
def query_graphrag(question: str) -> str:
    """Truy vấn đồ thị và trả lời câu hỏi."""
    try:
        # Sinh Cypher query
        cypher_query = cypher_chain.invoke(question)
        print(f"Cypher Query: {cypher_query}")

        # Thực thi query và trả về kết quả
        result = graph.query(cypher_query)
        return result
    except Exception as e:
        return f"Lỗi khi truy vấn: {e}"

if __name__ == "__main__":
    # Ví dụ truy vấn
    question = "What are the key factors for electric vehicle market growth?"
    answer = query_graphrag(question)
    print(f"Câu hỏi: {question}")
    print(f"Câu trả lời: {answer}")