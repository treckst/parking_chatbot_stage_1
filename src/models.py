from langchain_openai import OpenAIEmbeddings, ChatOpenAI

llm = ChatOpenAI(model='gpt-6-luna')

embedding = OpenAIEmbeddings(model='text-embedding-3-small')