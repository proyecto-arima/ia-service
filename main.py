import operator
import os
from typing import Annotated, List, Literal, TypedDict
import boto3
from langchain.chains.combine_documents.reduce import (acollapse_docs,
                                                       split_list_of_docs)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from pymongo import MongoClient
import requests

load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

token_max = 100000

# ---------------------------------------------- MONGODB UTILS ----------------------------------------------

def connect_to_mongodb():
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        client.admin.command("ping")
        print("Connected to MongoDB")
    except:
        print("Failed to connect to MongoDB")
        exit()
    return client

def get_raw_content(client):
    db = client['adaptaria']
    collection = db['contents']
    result = collection.find_one({"status": "PENDING"}, {"_id": 1, "key": 1}, sort=[('updatedAt', 1)])
    if result is None:
        print("No content found")
        client.close()
        exit(0)
    collection.update_one({"_id": result["_id"]}, {"$set": {"status": "PROCESSING"}})
    return {
        "id": result["_id"],
        "key": result["key"]
    }

def fetch_file_from_s3(key):
    s3_client = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    response = s3_client.generate_presigned_url('get_object', Params={'Bucket': os.getenv("BUCKET"), 'Key': f'{os.getenv("PREFIX")}/{key}'}, ExpiresIn=120)
    return response

def upload_generated_content(client, id, final_summary, markmap, game):
    db = client['adaptaria']
    collection = db['contents']
    collection.update_one({"_id": id}, {"$set": {"status": "DONE", "generated": [{ "type": "SUMMARY", "content": final_summary, "approved": False }, { "type": "MIND_MAP", "content": markmap, "approved": False }, { "type": "GAMIFICATION", "content": game, "approved": False}, { "type": "SPEECH", "content": "", "approved": False }]}})
    print("Content updated successfully")

# ------------------------------------------------------------------------------------------------------

# ---------------------------------------------- LOAD FILE ----------------------------------------------

def get_pages_from_url(url):
    loaded_file = PyPDFLoader(url)
    # file_path = os.path.basename(loaded_file.file_path)
    pages = loaded_file.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000000,
        chunk_overlap=10000,
        length_function=len
    ))
    return pages

# ------------------------------------------------------------------------------------------------------

# ---------------------------------------------- CHAT MODELS -------------------------------------------
# Uncomment the models that you want to use!

#OLLAMA_URL = os.getenv("OLLAMA_URL")

#llama_3_1_ollama = ChatOllama(model='llama3.1:70b-instruct-q2_K', base_url=OLLAMA_URL, verbose=True, disable_streaming=True)
#gemma2_llm_ollama = ChatOllama(model='gemma2:27b', base_url=OLLAMA_URL, temperature=3, num_predict=-2)
#deepseek_coder_v2 = Ollama(model='deepseek-coder-v2:16b', base_url=OLLAMA_URL, temperature=0, num_predict=-2)
#llama_3_1_ollama_for_code = Ollama(model='llama3.1:70b-instruct-q2_K', base_url=OLLAMA_URL, num_predict=-2, temperature=0)


# llama3.2:3b-instruct-fp16
#llama_3_2__instruct_ollama =  ChatOllama(model='llama3.2:3b-instruct-fp16', base_url=OLLAMA_URL, verbose=True, num_predict=-2, temperature=0.6)
#gemma2_instruct_llm_ollama =  ChatOllama(model='gemma2:27b-instruct-q2_K',  base_url=OLLAMA_URL, verbose=True, num_predict=-2, temperature=0.6)
#deepseek_coder_v2_instruct_ollama = ChatOllama(model='deepseek-coder-v2:16b-lite-base-fp16', base_url=OLLAMA_URL, verbose=True, num_predict=-2, temperature=0.6)

## Optionally you can use the semantic chunker to split the text into semantic chunks

#embeddings_engine = OllamaEmbeddings(model='llama3.1:70b-instruct-q2_K', base_url=OLLAMA_URL, mirostat=0, show_progress=True)
#semantic_chunker = SemanticChunker(embeddings_engine, breakpoint_threshold_type="gradient", sentence_split_regex=r"\.\n\n")
#pages = loaded_file.load_and_split(semantic_chunker)
#pages = [Document(page.page_content.replace("\n", " ")) for page in pages]

openai_gpt4 = ChatOpenAI(model='gpt-4o-mini', openai_api_key=os.getenv("OPENAI_API_KEY"), max_tokens=16384)

## Change the model to the one you want to use
summary_llm = openai_gpt4;
markmap_llm = openai_gpt4;
game_llm = openai_gpt4;

# ------------------------------------------------------------------------------------------------------


# ---------------------------------------------- PROMPTS -----------------------------------------------

map_prompt_template = """
Reescribe el siguiente texto de forma ligeramente resumida. Puedes obviar detalles irrelevantes o repetitivos.
Puedes reescribir el texto en un lenguaje más simple y claro, pero asegúrate de mantener la información principal.
No menciones que es un resumen y no incluyas texto del estilo: "El texto habla de..." o "En resumen,..."
No agregues información que no esté en el texto original.
Eres un proceso automático, tus respuestas no las va a leer un humano, no te disculpes. Si la respuesta no se puede generar, solo escribe "[<SKIPPED>]"
Si no se puede analizar el fragmento de texto por ser irrelevante, escribe "[<SKIPPED>]".

TEXTO A REESCRIBIR:
```
{context}
```
"""

reduce_to_markmap_template = """
Genera un mapa mental en formato MARKDOWN con los fragmentos de texto proporcionados.
Es sumamente importante tener en cuenta las siguientes consideraciones:
1. Debe ser en español. 
2. Debe respetar el formato MARKDOWN obligatoriamente. 
3. Debe tener los suficientes niveles de profundidad para que sea entendible. 
4. En cada nodo, agrega una pequeña descripción o explicación. 
5. Todos los nodos deben tener una explicación extraida del texto. 
6. No debes decir nada más que el markmap. 

Este es un ejemplo de MARKDOWN esperado:
```
- **Nodo 1**: Descripción del nodo 1
    - **Nodo 1.1**: Descripción del nodo 1.1
        - **Nodo 1.1.1**: Descripción del nodo 1.1.1
    - **Nodo 1.2**: Descripción del nodo 1.2
- **Nodo 2**: Descripción del nodo 2
    - **Nodo 2.1**: Descripción del nodo 2.1
    - **Nodo 2.2**: Descripción del nodo 2.2
```
FRAGMENTOS DE TEXTO:
```
{docs}
```
"""

reduce_to_summary_template = """
A continuación, se proveerán un conjunto de resúmenes de un texto.
Genera un texto unificado que contenga la información de todos los fragmentos proporcionados.
No menciones que es un resumen y no incluyas texto del estilo: "El texto habla de..." o "En resumen,...", solo escribe el contenido.
No agregues información que no esté en el texto original.
Eres un proceso automático, tus respuestas no las va a leer un humano, no te disculpes. Si la respuesta no se puede generar, solo escribe "[<SKIPPED>]"
Si no se puede resumir el fragmento de texto por ser irrelevante, ignoralo".

FRAGMENTOS DE TEXTO:
```
{docs}
```
"""

generate_game_template = """
Deberás generar un juego a partir de los fragmentos de texto proporcionados.
El juego consta de 3 niveles, cada uno con 3 preguntas.
Cada pregunta es un múltiple choise con 4 opciones.
Cada respuesta debe estar debidamente justificada.
El resultado debe ser un JSON obligatoriamente, sin texto extra.
El JSON esperado, debe tener un formato como el siguiente:
[
    {{
        "nivel": 1,
        "questions": [
            {{
                "text": "AQUÍ VA LA PREGUNTA",
                "options": [
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                ],
                "correctAnswer": "AQUÍ VA EL ÍNDICE DE LA RESPUESTA CORRECTA"
            }},
            {{
                "text": "AQUÍ VA LA PREGUNTA",
                "options": [
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                    {{
                        "answer": "AQUÍ VA UNA POSIBLE RESPUESTA",
                        "justification": "AQUÍ VA LA JUSTIFICACIÓN DE POR QUÉ LA RESPUESTA ES CORRECTA O INCORRECTA"
                    }},
                ],
                "correctAnswer": "AQUÍ VA EL ÍNDICE DE LA RESPUESTA CORRECTA"
            }}
        ]
    }}
    
]

FRAGMENTOS DE TEXTO:
{text}
"""

# -------------------------------------------------------------------------------------------------------

# ---------------------------------------------- Templates ----------------------------------------------

map_chat_prompt = ChatPromptTemplate.from_messages([("human", map_prompt_template)])
reduce_to_summary_prompt_chat = ChatPromptTemplate.from_messages([("human", reduce_to_summary_template)])
reduce_to_markmap = PromptTemplate(template=reduce_to_markmap_template, input_variables=["text"])
generate_game_prompt = ChatPromptTemplate.from_messages([("human", generate_game_template)])

# ------------------------------------------------------------------------------------------------------

# ---------------------------------------------- CHAINS ------------------------------------------------

map_chain_chat = map_chat_prompt | summary_llm | StrOutputParser()
markmap_chain = reduce_to_markmap | markmap_llm | StrOutputParser()
reduce_to_summary_chain_chat = reduce_to_summary_prompt_chat | summary_llm | StrOutputParser()
generate_game_chain = generate_game_prompt | game_llm | StrOutputParser()

# ------------------------------------------------------------------------------------------------------

def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(summary_llm.get_num_tokens(doc.page_content) for doc in documents)
  
  
# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str
    markmap: str
    game: str
    #trace_url: str
    #total_cost: float
    
    
# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str
    

# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    response = await map_chain_chat.ainvoke(state["content"])
    return {"summaries": [response]}
  
  
# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]
    
def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }
    
    
# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        docs = await acollapse_docs(doc_list, reduce_to_summary_chain_chat.ainvoke)
        print("DOCS: ", docs)
        results.append(docs)

    return {"collapsed_summaries": results}

# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"
      
      
# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    context = "".join([doc.page_content for doc in state["collapsed_summaries"]])
    response = await reduce_to_summary_chain_chat.ainvoke(context)
    return {"final_summary": response}
  
async def generate_markmap(state: OverallState):
    context = "".join([doc.page_content for doc in state["collapsed_summaries"]])
    response = await markmap_chain.ainvoke(context)
    return {"markmap": response}

async def generate_game(state: OverallState):
    context = "".join([doc.page_content for doc in state["collapsed_summaries"]])
    response = await generate_game_chain.ainvoke({"text": context})
    return {"game": response}

async def clean_markmap(state: OverallState):
    if state["markmap"].startswith("```") and not state["markmap"].startswith("```markdown"):
        return {"markmap": state["markmap"].removeprefix("```").removesuffix("```")}
    if state["markmap"].startswith("```markdown"):
        return {"markmap": state["markmap"].removeprefix("```markdown").removesuffix("```")}
    
    
async def clean_game(state: OverallState):
    if state["game"].startswith("```") and not state["game"].startswith("```json"):
        return {"game": state["game"].removeprefix("```").removesuffix("```")}
    if state["game"].startswith("```json"):
        return {"game": state["game"].removeprefix("```json").removesuffix("```")}
    
    
async def consolidate_results(state: OverallState):
    return {
        "final_summary": state["final_summary"],
        "markmap": state["markmap"],
        "game": state["game"]
    }
    
    


# def get_trace_url(state: OverallState):
#     API_URL_SESSIONS = 'https://api.smith.langchain.com/api/v1/sessions'
#     API_KEY = os.getenv('LANGCHAIN_API_KEY')
#     # Get the last session
#     response = requests.get(f'{API_URL_SESSIONS}?sort_by=start_time&sort_by_desc=true&limit=1', headers={'X-API-Key': API_KEY})
#     json_response = response.json()
#     id_session = json_response[0]['id']
#     total_cost = json_response[0]['total_cost']
#     print(id_session, total_cost)
#     API_URL_SHARE = f'https://api.smith.langchain.com/api/v1/runs/{id_session}/share'
#     response = requests.put(API_URL_SHARE, headers={'X-API-Key': API_KEY})
#     json_response = response.json()
#     print(json_response)
#     share_token = json_response['share_token']
#     return {"trace_url": f'https://smith.langchain.com/s/{share_token}', "total_cost": total_cost}
  
# Graph

# Construct the graph
# Nodes:
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_node("generate_markmap", generate_markmap)  # Nodo para generar markmap
graph.add_node("generate_game", generate_game)  # Nodo para generar juego
#graph.add_node("fetch_trace_url", get_trace_url)
graph.add_node("clean_markmap", clean_markmap)
graph.add_node("clean_game", clean_game)
graph.add_node("consolidate_results", consolidate_results)

# Edges:
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", "generate_markmap")  # Conectar resumen final con la generación del markmap
graph.add_edge("generate_markmap", "generate_game")  # Conectar resumen final con la generación del juego
graph.add_edge("generate_game", "clean_markmap")
graph.add_edge("clean_markmap", "clean_game")
graph.add_edge("clean_game", "consolidate_results")
graph.add_edge("consolidate_results", END)

#graph.add_edge("fetch_trace_url", END) 

app = graph.compile()
    

async def get_summaries(contents: List[str]):
    result = []
    async for step in app.astream({"contents": [doc.page_content for doc in pages]}, {"recursion_limit": 10},):
    #   print(list(step.keys()))
    #   print(step)
      result.append(step)
      
    return result[-1]['consolidate_results']
      
      
      
if __name__ == "__main__":
    import asyncio
    client = connect_to_mongodb()
    raw = get_raw_content(client)
    print(raw)
    url = fetch_file_from_s3(raw["key"])
    print(url)
    pages = get_pages_from_url(url)
    result = asyncio.run(get_summaries(pages), debug=True)
    upload_generated_content(client, raw["id"], result["final_summary"], result["markmap"], result["game"])
    client.close()
    
    
    
# from IPython.display import Image

# Image(app.get_graph().draw_mermaid_png())
