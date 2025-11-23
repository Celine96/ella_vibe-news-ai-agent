from openai import OpenAI

client = OpenAI(api_key="...")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["냥냥몬은 고양이 포켓몬입니다.", "울냥몬은 냥냥몬의 진화형입니다."]
)

print(response.data[0].embedding)  # Print the embedding vector

