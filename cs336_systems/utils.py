from dataclasses import dataclass, asdict
import json
from pathlib import Path
import pandas as pd


@dataclass
class BenchmarkRow():
    model_size: str
    batch_size: int

    vocab_size: int
    context_length: int

    mode: str # "forward / forward+backward"

    num_warmup_steps: int
    num_measure_steps: int

    mean_ms: str
    std_ms: str

    device: str



class BenchmarkReporter():

    def __init__(
            self,
            jsonl_path: str | Path,
            md_path: str | Path
        ):
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path)

        # Ensure directories exist
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, row: BenchmarkRow):

        # Append to jsonl
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")



    def render_markdown(self):
        # Write Markdown

        # Refresh markdown file
        # Load JSONL
        df = pd.read_json(self.jsonl_path, lines=True)
        with self.md_path.open('w') as f:
            f.write(df.to_markdown(index=False))



@dataclass
class AttentionRow():
    batch_size: int
    d_model: int
    seq_length: int
    
    f_avg_ms: str
    f_std_ms: str
    b_avg_ms: str
    b_std_ms: str

    mem_before_bwd_mb: str
    status: str # "ok" / "oom" / "error:<Type>"




class AttentionBenchmarkReporter():

    def __init__(
            self,
            jsonl_path: str | Path,
            md_path: str | Path
        ):
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path)

        # Ensure directories exist
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, row: AttentionRow):

        # Append to jsonl
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")



    def render_markdown(self):
        # Write Markdown

        # Refresh markdown file
        # Load JSONL
        df = pd.read_json(self.jsonl_path, lines=True)
        with self.md_path.open('w') as f:
            f.write(df.to_markdown(index=False))