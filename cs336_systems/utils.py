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

    impl: str # "compiled / eager"

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

    f_soln: str
    b_soln: str
    mem_soln: str

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



@dataclass
class FlashRow():
    seq_length: int
    d_model: int
    
    f_ms: str
    b_ms: str
    e2e_ms: str

    impl: str # "pytorch / triton"

    status: str # "ok" / "oom" / "error:<Type>"




class FlashBenchmarkReporter():

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

    def reset(self):
        # Delete if exists
        self.jsonl_path.unlink(missing_ok=True)

        # Recreate empty file
        self.jsonl_path.touch()
    
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