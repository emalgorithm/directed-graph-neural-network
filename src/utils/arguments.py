import argparse
import logging

parser = argparse.ArgumentParser("Directed Graph Neural Network")

### Dataset Args
parser.add_argument("--dataset", type=str, help="Name of dataset", default="chameleon")
parser.add_argument("--dataset_directory", type=str, help="Directory to save datasets", default="dataset")
parser.add_argument("--checkpoint_directory", type=str, help="Directory to save checkpoints", default="checkpoint")

### Preprocessing Args
parser.add_argument("--undirected", action="store_true", help="Whether to use undirected version of graph")
parser.add_argument("--self_loops", action="store_true", help="Whether to add self-loops to the graph")
parser.add_argument("--transpose", action="store_true", help="Whether to use transpose of the graph")

### Model Args
parser.add_argument("--model", type=str, help="Model type", default="gnn")
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=3)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.0)
parser.add_argument("--alpha", type=float, help="Direction convex combination params", default=0.5)
parser.add_argument("--learn_alpha", action="store_true")
parser.add_argument("--conv_type", type=str, help="DirGNN Model", default="dir-gcn")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--jk", type=str, choices=["max", "cat", None], default="max")

### Training Args
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0)
parser.add_argument("--num_epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=10)
parser.add_argument("--num_runs", type=int, help="Max number of runs", default=1)

### System Args
parser.add_argument("--use_best_hyperparams", action="store_true")
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument("--num_workers", type=int, help="Num of workers for the dataloader", default=0)
parser.add_argument("--log", type=str, help="Log Level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
parser.add_argument("--profiler", action="store_true")

args = parser.parse_args()
logger = logging.getLogger(__name__)
logger.setLevel(level=getattr(logging, args.log.upper(), None))
