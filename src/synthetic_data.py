import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# This script bootstraps your training by generating high-quality 
# prompt-difficulty pairs using OpenAI.

async def generate_synthetic_data(n=50):
    client = AsyncOpenAI()
    
    system_prompt = (
        "You are an expert in N8N automation. "
        "Generate a list of diverse user prompts that ask for an automation workflow. "
        "Include a mix of difficulty levels: easy (single webhook -> action), "
        "medium (logic/filtering), and hard (branching/merging/loops). "
        "Return JSON format: [{'prompt': '...', 'difficulty': 'easy'}, ...]"
    )
    
    print(f"Generating {n} synthetic scenarios...")
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {n} distinct N8N workflow requests."}
        ],
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        # Handle if GPT wraps it in a key like "scenarios" or just returns a list
        scenarios = data.get("scenarios", data.get("prompts", []))
        
        with open("src/data.json", "w") as f:
            json.dump(scenarios, f, indent=2)
            
        print(f"Successfully saved {len(scenarios)} scenarios to src/data.json")
    except Exception as e:
        print(f"Error parsing synthetic data: {e}")

if __name__ == "__main__":
    asyncio.run(generate_synthetic_data())