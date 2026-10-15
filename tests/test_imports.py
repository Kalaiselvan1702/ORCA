import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    print("Importing torch...")
    import torch
    print("Torch:", torch.__version__)
    
    print("Importing models...")
    from src.orca.models.orca_model import OrcaModel
    print("OrcaModel imported.")
    
    print("Importing trainer...")
    from src.orca.training.trainer import OrcaTrainer
    print("OrcaTrainer imported.")
    
    print("Importing dataset...")
    from src.orca.data.dataset import OrcaMultimodalDataset
    print("OrcaMultimodalDataset imported.")
    
    print("All imports succeeded!")
except Exception as e:
    import traceback
    traceback.print_exc()
