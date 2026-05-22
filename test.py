import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
from pathlib import Path

# 1. Load the specific env file
load_dotenv("info.env")

# 2. Pull the variable
api_key = os.getenv("SUPER_SECRET_API_KEY")

if not api_key:
    print("0 - API Key not found!")
else:
    print("1 - API Key loaded.")

client = OpenAI(api_key=api_key)

p=Path("AgentPersonas/ContextInt.md")
p = Path.cwd()
agentP = p / 'AgentPersonas'
T_1_fp = p / 'TrainingData'
eels6 = T_1_fp / 'spectrum_eels_multiline_6.png'
ContextInt = agentP / 'ContextInt.md'
DataValidator = agentP / 'DataValidator.md'
GraphVision = agentP / 'GraphVision.md'
Obisdianfactor = agentP / 'Obisdianfactor.md'

# files = T_1_fp.rglob('*.png')
# for f in files:
#     print(f)

# print(ContextInt.read_text())

"""
response = client.responses.create(
    model = "gpt-5.5-pro",
    input = "tell me about your persona",
    instructions=ContextInt.read_text(),
    max_output_tokens=1000
)
print(response.output_text)
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
eelsb64=encode_image(eels6)

response = client.responses.create(
    model = "gpt-5.5-pro",
    input = eelsb64,
    instructions=ContextInt.read_text(),
    max_output_tokens=1000
)