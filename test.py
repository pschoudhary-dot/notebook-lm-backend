from openai import OpenAI

client = OpenAI(
    base_url="https://api.sree.shop/v1",
    api_key="ddc-m4qlvrgpt1W1E4ZXc4bvm5T5Z6CRFLeXRCx9AbRuQOcGpFFrX2"
)

# models = client.models.list()
# print("\nModels:")
# for model in models:
#     print(model)


completion = client.chat.completions.create(
  model="deepseek-v3",
  messages=[
    # {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How are you Doing’?"}
  ]
)
print(completion.choices[0].message.content)