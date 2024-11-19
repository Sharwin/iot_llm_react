import uuid

from dotenv import load_dotenv

load_dotenv()

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta

from langsmith import Client
from tqdm.auto import tqdm

client = Client()
project_name = "langgraph-react-iot"
num_days = 30

tool_runs = client.list_runs(
    project_name=project_name,
    start_time=datetime.now() - timedelta(days=num_days),
)

data = []
futures: list[Future] = []
trace_cursor = 0
trace_batch_size = 50

tool_runs_by_parent = defaultdict(lambda: defaultdict(set))
with ThreadPoolExecutor(max_workers=2) as executor:
    for run in tqdm(tool_runs):
        tool_runs_by_parent[run.trace_id]["tools_involved"].add(run.name)

        if len(tool_runs_by_parent) % trace_batch_size == 0:
            if this_batch := list(tool_runs_by_parent.keys())[
                             trace_cursor : trace_cursor + trace_batch_size
                             ]:
                trace_cursor += trace_batch_size
                futures.append(
                    executor.submit(
                        client.list_runs,
                        project_name=project_name,
                        run_ids=this_batch,
                    )
                )
    if this_batch := list(tool_runs_by_parent.keys())[trace_cursor:]:
        futures.append(
            executor.submit(
                client.list_runs,
                project_name=project_name,
                run_ids=this_batch,
            )
        )

for future in tqdm(futures):
    root_runs = future.result()
    for root_run in root_runs:
        root_data = tool_runs_by_parent[root_run.id]
        data.append(
            {
                "run_id": root_run.id,
                "run_name": root_run.name,
                "run_type": root_run.run_type,
                "inputs": root_run.inputs,
                "outputs": root_run.outputs,
                "tools_involved": list(root_data["tools_involved"]),
                "prompt_tokens": root_run.prompt_tokens,
                "total_tokens": root_run.total_tokens,
                "events": list(root_run.events),
                "start_time": (root_run.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                "end_time": (root_run.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                "completion_tokens": root_run.completion_tokens,
                "total_cost": float(root_run.total_cost) if root_run.total_cost is not None else 0.0,
                "completion_cost": float(root_run.completion_cost)if root_run.completion_cost is not None else 0.0,
                "trace_id": str(root_run.trace_id)
            }
        )


import pandas as pd

df = pd.DataFrame(data)
df.head()
df.to_csv('tool_runs_data_3.csv', index=False)