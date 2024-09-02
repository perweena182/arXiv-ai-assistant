import arxiv
import tempfile
import aiofiles
import aiofiles.os
import asyncio
import requests
import io
import re
import logging
import chainlit as cl
from PyPDF2 import PdfMerger
from openai import AsyncAzureOpenAI
from chainlit.context import context
from chainlit.user_session import user_session
from aiohttp import ClientSession
from metadata_pipeline import daily_metadata_task
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pinecone
from azure.core.credentials import AzureKeyCredential
from knowledge_graph import RAG_Graph
import os
# from azure.ai.textanalytics import TextAnalyticsClient
# from langchain.embeddings import BaseEmbeddings

logging.basicConfig(filename='combined_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["GROQ_API_KEY"] = "<your-groq-api>"
os.environ["NEO4J_URI"] = "<your-neo4j-uri>"
os.environ["NEO4J_USERNAME"] = "<neo4j-username>"
os.environ["NEO4J_PASSWORD"] = "<neo4j-password>"

rag_graph = RAG_Graph()

daily_task_scheduled = False
def initialize_embeddings():
    """Initialize the Azure embedding model."""
    logger.info("Initializing Azure embeddings...")
    
    # Set up your Azure credentials
    endpoint = "<your-azure-endpoint>"
    api_key = "<your-azure-api>"
    model_name = "text-embedding-ada-002"  # Replace with the appropriate Azure model name if necessary
    embedding_model = AzureOpenAIEmbeddings(azure_endpoint=endpoint, api_key=api_key, model=model_name)
    return embedding_model

def initialize_vector_stores(embedding_model):
    """Initialize Pinecone vector stores for metadata and chunks."""
    logger.info("Initializing Pinecone vector stores...")
     
    metadata_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxi-rag-metadata")
    chunks_vector_store = PineconeVectorStore.from_existing_index(embedding=embedding_model, index_name="arxiv-rag-chunks")
    return metadata_vector_store, chunks_vector_store

def initialize_text_splitter():
    """Initialize the recursive character text splitter."""
    logger.info("Initializing text splitter...")
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )

async def send_actions():
    """Send action options to the user."""
    actions = [
        cl.Action(name="ask_followup_question", value="followup_question", description="Uses The Previously Retrieved Context", label="Ask a Follow-Up Question"),
        cl.Action(name="ask_new_question", value="new_question", description="Retrieves New Context", label="Ask a New Question About the Same Paper"),
        cl.Action(name="ask_about_new_paper", value="new_paper", description="Ask About A Different Paper", label="Ask About a Different Paper")
    ]
    await cl.Message(content="### Please Select One of the Following Options:", actions=actions).send()

@cl.on_stop
async def on_stop():
    """Handle session stop event to clean up tasks."""
    streaming_task = user_session.get('streaming_task')
    if streaming_task:
        streaming_task.cancel()
        await send_actions()
    user_session.set('streaming_task', None)
    logger.info("Session stopped and streaming task cleaned up.")

@cl.on_chat_start
async def main():
    """Main function to start the chat session."""
    global daily_task_scheduled
    
    if not daily_task_scheduled:
        asyncio.create_task(daily_metadata_task())
        daily_task_scheduled = True
    
    embedding_model = initialize_embeddings()
    metadata_vector_store, chunks_vector_store = initialize_vector_stores(embedding_model)
    text_splitter = initialize_text_splitter()

    user_session.set('embedding_model', embedding_model)
    user_session.set('metadata_vector_store', metadata_vector_store)
    user_session.set('chunks_vector_store', chunks_vector_store)
    user_session.set('text_splitter', text_splitter)
    user_session.set('current_document_id', None)

    text_content = """## Welcome to the arXiv Research Assistant

This system is designed to support students, researchers, and enthusiasts by providing real-time access to, and understanding of, the extensive research continually uploaded to arXiv.

With daily updates, it seamlessly integrates new papers, ensuring users always have the latest information at their fingertips.

### Instructions
1. **Enter the Title**: Start by entering the title of the research paper you wish to learn more about.
2. **Select a Paper**: Select a paper from the retrieved list by entering its corresponding number.
3. **Database Check**: The system will verify if the paper is already in the database.
   - If it exists, you'll be prompted to enter your question.
   - If it does not exist, the system will download the paper to the database and then prompt you to enter your question.
4. **Read the Answer**: After receiving the answer, you can:
   - Ask a follow-up question.
   - Ask a new question about the same paper.
   - Ask a new question about a different paper.

### Get Started
When you're ready, follow the first step below.
"""
    await cl.Message(content=text_content).send()
    await ask_initial_query()

async def ask_initial_query():
    """Prompt the user to enter the title of the research paper."""
    res = await cl.AskUserMessage(content="### Please Enter the Title of the Research Paper You Wish to Learn More About:", timeout=3600).send()
    if res:
        initial_query = res['output']
        logger.info(f"Initial query received: {initial_query}")
        
        metadata_vector_store = user_session.get('metadata_vector_store')
        if not metadata_vector_store:
            logger.error("Metadata vector store not initialized.")
            return
        logger.info(f"Searching for metadata with query: {initial_query}")
        search_results = metadata_vector_store.similarity_search(query=initial_query, k=5)
        logger.info(f"Metadata search results: {search_results}")
        if not search_results:
            await cl.Message(content="No Search Results Found").send()
            return
        selected_doc_id = await select_document_from_results(search_results)
        if selected_doc_id:
            logger.info(f"Document selected with ID: {selected_doc_id}")
            user_session.set('current_document_id', selected_doc_id)
            await process_and_upload_chunks(selected_doc_id)
       

async def ask_user_question(document_id):
    """Prompt the user to enter a question about the selected document."""
    logger.info(f"Asking user question for document_id: {document_id}")
    context, user_query = await process_user_query(document_id)
    if context and user_query:
        task = asyncio.create_task(query_openai_with_context(context, user_query))
        user_session.set('streaming_task', task)
        await task
async def select_document_from_results(search_results):
    """Allow user to select a document from the search results."""
    if not search_results:
        await cl.Message(content="No Search Results Found").send()
        return None

    message_content = "### Please Enter the Number Corresponding to Your Desired Paper:\n"
    message_content += "| No. | Paper Title | Doc. ID |\n"
    message_content += "|-----|-------------|---------|\n"

    for i, doc in enumerate(search_results, start=1):
        page_content = doc.page_content
        document_id = doc.metadata['document_id']
        message_content += f"| {i} | {page_content} | {document_id} |\n"

    await cl.Message(content=message_content).send()

    while True:
        res = await cl.AskUserMessage(content="", timeout=3600).send()
        if res:
            try:
                user_choice = int(res['output']) - 1
                if 0 <= user_choice < len(search_results):
                    selected_doc_id = search_results[user_choice].metadata['document_id']
                    selected_paper_title = search_results[user_choice].page_content
                    await cl.Message(content=f"\n**You selected:** {selected_paper_title}").send()
                    return selected_doc_id
                else:
                    await cl.Message(content="\nInvalid Selection. Please enter a valid number from the list.").send()
            except ValueError:
                await cl.Message(content="\nInvalid input. Please enter a number.").send()
        else:
            await cl.Message(content="\nNo selection made. Please enter a valid number from the list.").send()

async def do_chunks_exist_already(document_id):
    """Check if chunks for the document already exist."""
    chunks_vector_store = user_session.get('chunks_vector_store')
    filter = {"document_id": {"$eq": document_id}}
    test_query = chunks_vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    logger.info(f"Chunks existence check result for document_id {document_id}: {test_query}")
    return bool(test_query)

async def download_pdf(session, document_id, url, filename):
    """Download the PDF file asynchronously."""
    logger.info(f"Downloading PDF for document_id: {document_id} from URL: {url}")
    async with session.get(url) as response:
        if response.status == 200:
            async with aiofiles.open(filename, mode='wb') as f:
                await f.write(await response.read())
            logger.info(f"Successfully downloaded PDF for document_id: {document_id}")
        else:
            logger.error(f"Failed to download PDF for document_id: {document_id}, status code: {response.status}")
            raise Exception(f"Failed to download PDF: {response.status}")

async def extract_arxiv_ids_from_text(references_text):
    """Extract arXiv IDs from a block of text."""
    arxiv_ids = set()  # Use a set to store unique arXiv IDs
    pattern = r'arXiv:\d{4}\.\d{5}'  # Regular expression to match arXiv IDs

    # Find all arXiv IDs in the references text
    matches = re.findall(pattern, references_text)
    arxiv_ids.update(matches)  # Add matches to the set

    return arxiv_ids

async def generate_arxiv_links(arxiv_ids):
    """Generate arXiv links from a set of arXiv IDs."""
    base_url = "https://arxiv.org/abs/"
    return [f"{base_url}{arxiv_id.split(':')[1]}" for arxiv_id in arxiv_ids]
async def download(arxiv_id):
    """Download the PDF of the arXiv paper by its ID."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    return io.BytesIO(response.content)

async def merge_pdfs(pdf_streams):
    """Merge multiple PDFs into a single in-memory PDF."""
    merger = PdfMerger()
    for pdf in pdf_streams:
        merger.append(pdf)
    merged_pdf = io.BytesIO()
    merger.write(merged_pdf)
    merged_pdf.seek(0)  # Move the pointer to the beginning of the stream
    return merged_pdf


async def process_and_upload_chunks(document_id):
    """Download, process, and upload chunks of the document."""
    await cl.Message(content="#### It seems that paper isn't currently in our database. Don't worry, we are currently downloading, processing, and uploading it. This will only take a few moments.").send()
    await asyncio.sleep(2)

    try:
        async with ClientSession() as session:
            paper = await asyncio.to_thread(next, arxiv.Client().results(arxiv.Search(id_list=[str(document_id)])))
            url = paper.pdf_url
            filename = f"{document_id}.pdf"
            await download_pdf(session, document_id, url, filename)

        loader = PyPDFLoader(filename)
        pages = await asyncio.to_thread(loader.load)

        text_splitter = user_session.get('text_splitter')
        content = []
        references_text = ""
        found_references = False
        for page in pages:
            page_text = page.page_content
            if found_references:
                references_text += page_text
            elif "references" in page_text.lower():
                content.append(page_text.split("References")[0])
                references_text += page_text.split("References")[1] if len(page_text.split("References")) > 1 else ""
                found_references = True
            else:
                content.append(page_text)

        full_content = ''.join(content)
        pdf_streams = []
        
        if found_references:
            arxiv_ids =list(await extract_arxiv_ids_from_text(references_text))
            max_ids = 5
            limited_arxiv_ids = arxiv_ids[:max_ids]
        
            if limited_arxiv_ids:
                print("Extracted arXiv IDs:")
                for arxiv_id in limited_arxiv_ids:
                    print(arxiv_id)
                    try:
                        pdf_stream = await download(arxiv_id)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(pdf_stream.getvalue())
                            pdf_stream_path = temp_file.name
                        pdf_streams.append(pdf_stream_path)
                    except requests.HTTPError as e:
                        print(f"Failed to download PDF for arXiv ID {arxiv_id}: {e}")
                    #     pdf_streams.append(pdf_stream)
                    # except requests.HTTPError as e:
                    #     print(f"Failed to download PDF for arXiv ID {arxiv_id}: {e}")
                
            else:
                print("No arXiv IDs found.")
        if pdf_streams:
            merged_pdf_stream =await merge_pdfs(pdf_streams)
            print("Merged PDF created successfully.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(merged_pdf_stream.getvalue())
                merged_pdf_path = temp_file.name

            # Load the merged PDF
            loader = PyPDFLoader(merged_pdf_path)
            
            # Step 4: Load the merged PDF as raw documents using PyPDFLoader
            # loader = PyPDFLoader(merged_pdf_stream)
            raw_documents = loader.load()
            if isinstance(raw_documents, list):
    # Join list elements to form a single string
                raw_content = ''.join([str(doc) for doc in raw_documents])
            else:
            # If raw_documents is already a string
                raw_content = str(raw_documents)

        # Combine full_content and raw_content
            final_content = full_content + raw_content
            # rag_graph.create_graph(final_content)
        
        else:
            final_content=full_content
            rag_graph.create_graph(final_content)

        embedding_model = user_session.get('embedding_model')
        if not embedding_model:
            raise ValueError("Embedding model not initialized")


        chunks = text_splitter.split_text(final_content)
        chunks_vector_store = user_session.get('chunks_vector_store')
        await asyncio.to_thread(
            chunks_vector_store.from_texts,
            texts=chunks,
            embedding=embedding_model,
            metadatas=[{"document_id": document_id} for _ in chunks],
            index_name="arxiv-rag-chunks"
        )

        await aiofiles.os.remove(filename)
        if pdf_streams:
            for pdf_path in pdf_streams:
                await aiofiles.os.remove(pdf_path)
        if 'merged_pdf_path' in locals():
            await aiofiles.os.remove(merged_pdf_path)
        logger.info(f"Successfully processed and uploaded chunks for document_id: {document_id}")
        await ask_user_question(document_id)



    except Exception as e:
        logger.error(f"Error processing and uploading chunks for document_id {document_id}: {e}")
        await cl.Message(content="#### An error occurred during processing. Please try again.").send()
        return

async def retrieve_vector_context(chunks_vector_store, user_query, document_id):
    context = []
    filter = {"document_id": {"$eq": document_id}}
    attempts = 5

    for attempt in range(attempts):
        search_results = chunks_vector_store.similarity_search(query=user_query, k=15, filter=filter)
        logging.info(f"Context retrieval attempt {attempt + 1}: Found {len(search_results)} results")
        context = [doc.page_content for doc in search_results]
        if context:
            break
        logging.info(f"No context found, retrying... (attempt {attempt + 1}/{attempts})")
        await asyncio.sleep(2)
    
    logging.info(f"User query processed. Context length: {len(context)}, User Query: {user_query}")
    return context

async def process_user_query(document_id):
    """Process the user's query about the document."""
    res = await cl.AskUserMessage(content="### Please Enter Your Question:", timeout=3600).send()
    if res:
        user_query = res['output']
        # context = []
        chunks_vector_store = user_session.get('chunks_vector_store')
        vector_context = await retrieve_vector_context(chunks_vector_store, user_query, document_id)
        graph_context = rag_graph.ask_question_chain(user_query)
        if isinstance(vector_context, str):
            vector_context = [vector_context]

        if isinstance(graph_context, str):
            graph_context = [graph_context]
        print(f"graph_context: {graph_context}")

    # Combine both contexts
        context = vector_context +graph_context

        

        # logger.info(f"User query processed. Context length: {len(context)}, User Query: {user_query}")
        return context, user_query
    return None, None

async def query_openai_with_context(context, user_query):
    """Query OpenAI with the context and user query."""
    if not context:
        await cl.Message(content="No context available to answer the question.").send()
        return

    client = AsyncAzureOpenAI(
        azure_endpoint="<your-azure-endpoint>",
        api_key="<your-api-key>",  # Replace with your actual API key
        api_version="2023-05-15")

    settings = {
        "model": "gpt-35-turbo-16k",
        "temperature": 0.3,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    message_history = [
        {"role": "system", "content": """
         Your job is to answer the user's query using only the provided context.
         Be detailed and long-winded. Format your responses in markdown formatting, making good use of headings,
         subheadings, ordered and unordered lists, and regular text formatting such as bolding of text and italics.
         Sometimes the equations retrieved from the context will be formatted improperly and in an incompatible format
         for correct LaTeX rendering. Therefore, if you ever need to provide equations, make sure they are
         formatted properly using LaTeX, wrapping the equation in single dollar signs ($) for inline equations
         or double dollar signs ($$) for bigger, more visual equations. Keep your answer grounded in the facts
         of the provided context. If the context does not contain the facts needed to answer the user's query, return:
         "I do not have enough information available to accurately answer the question."
         """},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Question: {user_query}"}
    ]

    msg = cl.Message(content="")
    await msg.send()

    async def stream_response():
        stream = await client.chat.completions.create(messages=message_history, stream=True, **settings)
        async for part in stream:
            if token := part.choices[0].delta.content:
                await msg.stream_token(token)

    streaming_task = asyncio.create_task(stream_response())
    user_session.set('streaming_task', streaming_task)

    try:
        await streaming_task
    except asyncio.CancelledError:
        streaming_task.cancel()
        return

    await msg.update()
    await send_actions()

@cl.action_callback("ask_followup_question")
async def handle_followup_question(action):
    """Handle follow-up question action."""
    logger.info("Follow-up question button clicked.")
    current_document_id = user_session.get('current_document_id')
    if current_document_id:
        context, user_query = await process_user_query(current_document_id)
        if context and user_query:
            logger.info(f"Processing follow-up question for document_id: {current_document_id}")
            task = asyncio.create_task(query_openai_with_context(context, user_query))
            user_session.set('streaming_task', task)
            await task
        else:
            logger.warning("Context or user query not found for follow-up question.")
    else:
        logger.warning("No current document ID found for follow-up question.")

@cl.action_callback("ask_new_question")
async def handle_new_question(action):
    """Handle new question action."""
    logger.info("New question about the same paper button clicked.")
    current_document_id = user_session.get('current_document_id')
    if current_document_id:
        logger.info(f"Asking new question for document_id: {current_document_id}")
        await ask_user_question(current_document_id)
    else:
        logger.warning("No current document ID found for new question.")

@cl.action_callback("ask_about_new_paper")
async def handle_new_paper(action):
    """Handle new paper action."""
    logger.info("New paper button clicked.")
    await ask_initial_query()

if __name__ == "__main__":
    asyncio.run(main())