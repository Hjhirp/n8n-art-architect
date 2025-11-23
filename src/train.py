import unsloth # FIXED: Must be the very first import to patch optimization
import os
import json
import asyncio
import statistics
import art
# FIXED: Explicit import needed for LocalBackend
from art.local import LocalBackend 
from datetime import datetime
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import AsyncOpenAI

# Load Environment Variables
load_dotenv()

# --- Configuration ---
PROJECT_NAME = "n8n-workflow-architect"
MODEL_NAME = "n8n-architect-14b-v1"
# CHANGED: Switched to 14B model. This fits comfortably on A100 40GB.
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"

# MCP Configuration
MCP_CMD = "npx"
MCP_ARGS = ["-y", "@modelcontextprotocol/server-n8n"] 

# --- A100 40GB TUNING ---
# With 14B, we can safely increase context to 8192 tokens
MAX_CONTEXT_LEN = 8192 
# We can use more GPU memory for caching since the weights are smaller (~9GB)
GPU_UTILIZATION = 0.8 

# Training Config
TRAIN_EPOCHS = 1
TRAIN_STEPS = 10
LEARNING_RATE = 2e-5

# --- Data Structures ---
@dataclass
class Scenario:
    id: str
    prompt: str
    difficulty: str

@dataclass
class EvalMetric:
    syntax_score: float    # 0.0 - 0.5 (MCP Validation)
    semantic_score: float  # 0.0 - 0.5 (LLM Judge)
    total_score: float     # Sum
    is_valid_json: bool

# --- 1. Validation & Scoring Logic ---
async def score_workflow(prompt: str, generated_text: str) -> EvalMetric:
    metric = EvalMetric(0.0, 0.0, 0.0, False)
    workflow_json = {}
    
    # A. Extract JSON
    try:
        if "```json" in generated_text:
            json_str = generated_text.split("```json")[1].split("```")[0]
        elif "{" in generated_text:
            start = generated_text.find("{")
            end = generated_text.rfind("}") + 1
            json_str = generated_text[start:end]
        else:
            json_str = generated_text
        
        workflow_json = json.loads(json_str)
        metric.is_valid_json = True
    except (json.JSONDecodeError, ValueError):
        return metric 

    # B. MCP Validation (Syntax)
    try:
        server_params = StdioServerParameters(command=MCP_CMD, args=MCP_ARGS)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    res = await session.call_tool("validate_workflow", {"workflow": workflow_json})
                    content = res.content[0].text.lower()
                    
                    if "valid" in content and "error" not in content:
                        metric.syntax_score = 0.5 
                    elif "warning" in content:
                        metric.syntax_score = 0.25
                    else:
                        metric.syntax_score = 0.0
                except Exception:
                    if "nodes" in workflow_json and "connections" in workflow_json:
                        metric.syntax_score = 0.1
    except Exception as e:
        print(f"[MCP Connection Error] {e}")

    # C. RULER Judge (Semantics)
    if metric.is_valid_json:
        try:
            client = AsyncOpenAI()
            judge_res = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an N8N QA Lead. Rate logic 0.0 to 0.5 based on User Goal. Return float only."},
                    {"role": "user", "content": f"User Goal: {prompt}\n\nWorkflow JSON: {str(workflow_json)[:MAX_CONTEXT_LEN]}..."}
                ]
            )
            metric.semantic_score = float(judge_res.choices[0].message.content.strip())
        except Exception:
            metric.semantic_score = 0.0

    metric.total_score = metric.syntax_score + metric.semantic_score
    return metric

# --- 2. Rollout Wrapper ---
async def rollout(model: art.Model, scenario: Scenario):
    messages = [
        {"role": "system", "content": "You are an expert N8N Architect. Output valid JSON."},
        {"role": "user", "content": f"Create workflow: {scenario.prompt}"}
    ]
    
    response = await model.generate(
        messages=messages,
        max_tokens=4096, # Increased generation limit for larger workflows
        temperature=0.8
    )
    
    generated_text = response.choices[0].message.content
    metrics = await score_workflow(scenario.prompt, generated_text)
    
    traj = art.Trajectory(
        messages_and_choices=messages,
        metadata={"prompt": scenario.prompt, "id": scenario.id},
        reward=metrics.total_score
    )
    traj.messages_and_choices.append(response.choices[0])
    return traj.finish()

# --- 3. Evaluation Loop ---
async def run_evaluation_set(model: art.Model, scenarios: list[Scenario], stage: str):
    print(f"\n>>> Running {stage.upper()} Evaluation <<<")
    results = []
    
    for i, scen in enumerate(scenarios):
        print(f"Eval {i+1}/{len(scenarios)}...")
        messages = [
            {"role": "system", "content": "You are an expert N8N Architect. Output valid JSON."},
            {"role": "user", "content": f"Create workflow: {scen.prompt}"}
        ]
        res = await model.generate(messages=messages, max_tokens=4096, temperature=0.0)
        text = res.choices[0].message.content
        
        metric = await score_workflow(scen.prompt, text)
        results.append({"metrics": asdict(metric)})
    
    if not results: return {}

    avg_total = statistics.mean([r["metrics"]["total_score"] for r in results])
    avg_valid = statistics.mean([1.0 if r["metrics"]["is_valid_json"] else 0.0 for r in results]) * 100
    
    summary = {
        "stage": stage,
        "avg_total_score": avg_total,
        "valid_json_percent": avg_valid,
        "detailed_results": results
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_{stage}.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"--- {stage.upper()} Avg Score: {avg_total:.3f} | Valid JSON: {avg_valid:.1f}% ---")
    return summary

# --- 4. Synthetic Data (Mini) ---
def get_scenarios():
    prompts = [
        ("Webhook to Slack", "easy"),
        ("Google Sheets to Email", "medium"),
        ("Typeform to Airtable", "medium"),
        ("Cron to Postgres", "medium"),
        ("Webhook to Discord", "easy"),
        ("Shopify to Invoice", "hard")
    ]
    return [Scenario(f"s_{i}", p, diff) for i, (p, diff) in enumerate(prompts)]

# --- 5. Main Pipeline ---
async def main():
    print(f"Initializing Model: {BASE_MODEL}")
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL 
    )
    
    model._internal_config = art.dev.InternalModelConfig(
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=GPU_UTILIZATION,
            max_model_len=MAX_CONTEXT_LEN
        )
    )
    # FIXED: Use the directly imported class
    backend = LocalBackend(in_process=True) 
    await model.register(backend)

    all_scenarios = get_scenarios()
    train_set = all_scenarios[:4]
    test_set = all_scenarios[4:]

    pre_stats = await run_evaluation_set(model, test_set, "before")

    print("\n>>> Training Loop <<<")
    training_config = art.TrainConfig(learning_rate=LEARNING_RATE, num_epochs=TRAIN_EPOCHS)
    
    for step in range(TRAIN_STEPS):
        print(f"Step {step+1}/{TRAIN_STEPS}")
        groups = []
        for scen in train_set: 
            trajectories = [await rollout(model, scen) for _ in range(2)]
            groups.append(art.TrajectoryGroup(trajectories))
        
        await model.train(groups, config=training_config)

    post_stats = await run_evaluation_set(model, test_set, "after")

    final_report = {
        "improvement_score": post_stats["avg_total_score"] - pre_stats["avg_total_score"],
        "validity_lift": post_stats["valid_json_percent"] - pre_stats["valid_json_percent"]
    }
    
    with open("results/final_comparison.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nDONE. Improvement: {final_report['improvement_score']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())