import openai

# Set your API key
openai.api_key = '<api-key>'

#

# new
from openai import OpenAI

client = OpenAI(
  api_key='<openai_api_key>'  # this is also the default, it can be omitted
)

# Example text
text = "OpenAI provides powerful language models."

# Request embeddings
# client.embeddings.create()
response = client.embeddings.create(input=text, model="text-embedding-ada-002")

# Print embeddings
embeddings = response['data'][0]['embedding']
print(embeddings)
