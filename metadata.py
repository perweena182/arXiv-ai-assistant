import aiofiles
import os
import aiofiles.os
import aiofiles.ospath
import asyncio
import logging
import pandas as pd
from sickle import Sickle
from sickle.oaiexceptions import NoRecordsMatch
from requests.exceptions import HTTPError, RequestException
from datetime import datetime, timedelta
import pytz
import xml.etree.ElementTree as ET
import ast
import time
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_metadata(from_date, until_date):
    connection = Sickle('http://export.arxiv.org/oai2')
    logging.info('Getting papers...')
    params = {'metadataPrefix': 'arXiv', 'from': from_date, 'until': until_date, 'ignore_deleted': True}
    data = connection.ListRecords(**params)
    logging.info('Papers retrieved.')

    iters = 0
    errors = 0

    with open('arXiv_metadata_raw.xml', 'a+', encoding="utf-8") as f:
        while True:
            try:
                record = next(data).raw
                f.write(record)
                f.write('\n')
                errors = 0
                iters += 1
                if iters % 1000 == 0:
                    logging.info(f'{iters} Processing Attempts Made Successfully.')

            except HTTPError as e:
                handle_http_error(e)

            except RequestException as e:
                logging.error(f'RequestException: {e}')
                raise

            except StopIteration:
                logging.info(f'Metadata For The Specified Period, {from_date} - {until_date} Downloaded.')
                break

            except Exception as e:
                errors += 1
                logging.error(f'Unexpected error: {e}')
                if errors > 5:
                    logging.critical('Too many consecutive errors, stopping the harvester.')
                    raise

def handle_http_error(e):
    if e.response.status_code == 503:
        retry_after = e.response.headers.get('Retry-After', 30)
        logging.warning(f"HTTPError 503: Server busy. Retrying after {retry_after} seconds.")
        time.sleep(int(retry_after))
    else:
        logging.error(f'HTTPError: Status code {e.response.status_code}')
        raise e

def parse_xml_to_df(xml_file):
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    
    if not xml_content.strip().startswith('<root>'):
        xml_content = f"<root>{xml_content}</root>"

    root = ET.ElementTree(ET.fromstring(xml_content)).getroot()
    records = []
    ns = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'arxiv': 'http://arxiv.org/OAI/arXiv/'
    }
    
    for record in root.findall('oai:record', ns):
        data = {}
        header = record.find('oai:header', ns)
        data['identifier'] = header.find('oai:identifier', ns).text
        data['datestamp'] = header.find('oai:datestamp', ns).text
        data['setSpec'] = [elem.text for elem in header.findall('oai:setSpec', ns)]
        
        metadata = record.find('oai:metadata/arxiv:arXiv', ns)
        data['id'] = metadata.find('arxiv:id', ns).text
        data['created'] = metadata.find('arxiv:created', ns).text
        data['updated'] = metadata.find('arxiv:updated', ns).text if metadata.find('arxiv:updated', ns) is not None else None
        data['authors'] = [
            (author.find('arxiv:keyname', ns).text if author.find('arxiv:keyname', ns) is not None else None,
             author.find('arxiv:forenames', ns).text if author.find('arxiv:forenames', ns) is not None else None)
            for author in metadata.findall('arxiv:authors/arxiv:author', ns)
        ]
        data['title'] = metadata.find('arxiv:title', ns).text
        data['categories'] = metadata.find('arxiv:categories', ns).text
        data['comments'] = metadata.find('arxiv:comments', ns).text if metadata.find('arxiv:comments', ns) is not None else None
        data['report_no'] = metadata.find('arxiv:report-no', ns).text if metadata.find('arxiv:report-no', ns) is not None else None
        data['journal_ref'] = metadata.find('arxiv:journal-ref', ns).text if metadata.find('arxiv:journal-ref', ns) is not None else None
        data['doi'] = metadata.find('arxiv:doi', ns).text if metadata.find('arxiv:doi', ns) is not None else None
        data['license'] = metadata.find('arxiv:license', ns).text if metadata.find('arxiv:license', ns) is not None else None
        data['abstract'] = metadata.find('arxiv:abstract', ns).text.strip() if metadata.find('arxiv:abstract', ns) is not None else None
        
        records.append(data)
    df = pd.DataFrame(records)
    return df

def preprocess_dataframe(df):
    df = df[['datestamp', 'id', 'created', 'authors', 'title', 'abstract','journal_ref']].copy()
    
    df.rename(columns={
        'datestamp': 'last_edited',
        'id': 'document_id',
        'created': 'date_created'
    }, inplace=True)
    
    df.loc[:, 'title'] = df['title'].astype(str)
    df.loc[:, 'authors'] = df['authors'].astype(str)
    df.loc[:, 'abstract'] = df['abstract'].astype(str)
    df.loc[:, 'journal_ref'] = df['journal_ref'].astype(str)
    
    df.loc[:, 'title'] = df['title'].str.replace('  ', ' ', regex=True)
    df.loc[:, 'authors'] = df['authors'].str.replace('  ', ' ', regex=True)
    df.loc[:, 'abstract'] = df['abstract'].str.replace('  ', ' ', regex=True)
    df.loc[:, 'journal_ref'] = df['journal_ref'].str.replace('  ', ' ', regex=True)
    
    
    df.loc[:, 'title'] = df['title'].str.replace('\n', '', regex=True)
    df.loc[:, 'abstract'] = df['abstract'].str.replace('\n', '', regex=True)
    df.loc[:, 'journal_ref'] = df['journal_ref'].str.replace(' \n', '', regex=True)
    
    
    df.loc[:, 'authors'] = df['authors'].str.replace('[\[\]\'"()]', '', regex=True)

    def flip_names(authors):
        author_list = authors.split(', ')
        flipped_authors = []
        for i in range(0, len(author_list), 2):
            if i+1 < len(author_list):
                flipped_authors.append(f"{author_list[i+1]} {author_list[i]}")
        return ', '.join(flipped_authors)

    df.loc[:, 'authors'] = df['authors'].apply(flip_names)
    
    df.loc[:, 'last_edited'] = pd.to_datetime(df['last_edited'])
    df.loc[:, 'date_created'] = pd.to_datetime(df['date_created'])
    
    df = df[df['document_id'].str.match('^\d')]

    df = df[df['last_edited'] == df['date_created'] + pd.Timedelta(days=1)]
    
    df.loc[:, 'title_by_authors'] = df['title'] + ' by ' + df['authors']
    
    df.drop(['title', 'authors', 'date_created', 'last_edited'], axis=1, inplace=True)
    
    df.to_csv('metadata_processed_4.csv', index=False)
    return df
def upload_to_pinecone(df, vector_store):
    texts = df['title_by_authors'].tolist()
    metadatas = df[['document_id','abstract']].to_dict(orient='records')
    
    logging.info(f"Uploading {len(texts)} records to Pinecone.")
    try:
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        logging.info("Upload successful.")
    except Exception as e:
        logging.error(f"Failed to upload to Pinecone: {e}")

# def upload_to_pinecone(df, vector_store):
#     texts = df['title_by_authors'].tolist()
#     metadatas = df[['document_id']].to_dict(orient='records')
#     vector_store.add_texts(texts=texts, metadatas=metadatas)

def setup_logging():
    logging.basicConfig(filename='arxiv_download.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    setup_logging()
    from_date = '2024-08-01'
    until_date = '2024-08-01'
    download_metadata(from_date, until_date)
    xml_file = 'arXiv_metadata_raw.xml'
    df = parse_xml_to_df(xml_file)
    os.remove(xml_file)
    df = preprocess_dataframe(df)
    if not df.empty:
        PINECONE_API_KEY = "1ef1bebd-a031-4f0c-b61d-8b0b117e6665"
        endpoint = "https://botgpt4.openai.azure.com/"
        api_key = "937ecf0c2d0c4191892ac6187c1c06f1"
        model_name = "text-embedding-ada-002"  # Replace with the appropriate Azure model name if necessary
        embedding_model = AzureOpenAIEmbeddings(azure_endpoint=endpoint, api_key=api_key, model=model_name)
        index_name = "arxi-rag-metadata"
        vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
        upload_to_pinecone(df, vector_store)
    else:
        logging.error("DataFrame is empty. Skipping upload.")
if __name__ == "__main__":
    asyncio.run(main())