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
ContextInt = agentP / 'ContextInt.md'
DataValidator = agentP / 'DataValidator.md'
GraphVision = agentP / 'GraphVision.md'
Obisdianfactor = agentP / 'Obisdianfactor.md'