import json
from pathlib import Path

def create_nb(cells, target_path):
    nb = {
        'cells': cells,
        'metadata': {
            'language_info': {'name': 'python', 'version': '3.13.1'}
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Created {target_path}")

# 01_eda.ipynb
c1 = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# ORCA: Exploratory Data Analysis (EDA)\n',
            'This notebook explores the multimodal telemetry from the RCAEval benchmark: synchronized logs, Prometheus metrics, and distributed OpenTelemetry traces across Google Online Boutique microservices.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os\n',
            'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"\n',
            'import pandas as pd\n',
            'import numpy as np\n',
            'import matplotlib.pyplot as plt\n',
            'from pathlib import Path\n',
            'from src.orca.preprocessing.normalizer import SERVICES, CANONICAL_METRICS\n',
            'print(f"Target microservices ({len(SERVICES)}):", SERVICES)\n',
            'print(f"Canonical metrics count: {len(CANONICAL_METRICS)}")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Inspect processed dataset summary\n',
            'import json\n',
            'with open("data/processed/dataset_summary.json", "r") as f:\n',
            '    summary = json.load(f)\n',
            'print("Processed dataset summary:")\n',
            'print(json.dumps(summary, indent=2))\n'
        ]
    }
]
create_nb(c1, 'notebooks/01_eda.ipynb')

# 02_model_pipeline.ipynb
c2 = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# ORCA: Neural Model Architecture & Forward Pass\n',
            'Demonstration of the deep multimodal architecture combining Log Transformer, Metric TCN-GRU, GATv2 Graph Neural Network, and Cross-Attention Fusion.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os\n',
            'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"\n',
            'import torch\n',
            'from src.orca.models.orca_model import OrcaModel\n',
            'print("PyTorch version:", torch.__version__)\n',
            'model = OrcaModel(\n',
            '    vocab_size=1000, log_embed_dim=64, log_hidden_dim=128, log_num_layers=2, log_heads=4,\n',
            '    in_metrics=70, metric_hidden_dim=128, metric_num_layers=2,\n',
            '    node_feat_dim=27, graph_hidden_dim=128, graph_num_layers=2, graph_heads=2,\n',
            '    fusion_type="cross_attention", hidden_dim=128, num_services=10\n',
            ')\n',
            'total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n',
            'print(f"Total trainable parameters: {total_params}")\n'
        ]
    }
]
create_nb(c2, 'notebooks/02_model_pipeline.ipynb')

# 03_evaluation_demo.ipynb
c3 = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# ORCA: Evaluation & Diagnostic Demo\n',
            'Demonstrates loading the trained model checkpoint, running inference on incident test windows, and visualizing root-cause localization probabilities.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os\n',
            'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"\n',
            'import json\n',
            'import pandas as pd\n',
            'from app.inference import OrcaInferenceEngine\n',
            'from app.services import IncidentDiagnosisService\n',
            'from app.schemas import IncidentPayload\n',
            '\n',
            'engine = OrcaInferenceEngine("artifacts/models/orca_dev_checkpoint.pt")\n',
            'svc = IncidentDiagnosisService(engine)\n',
            'print("Model Engine and Service Initialized successfully!")\n'
        ]
    }
]
create_nb(c3, 'notebooks/03_evaluation_demo.ipynb')
print("All notebooks created successfully!")
