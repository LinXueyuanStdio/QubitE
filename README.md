# QubitE

code for "QubitE: Qubit Embedding for Knowledge Graph Completion".

## Environment

- PyTorch 1.8.1 + cuda 10.2

## How to Run

QubitE with $\psi != 0$ :
```shell
python train_QubitE_best_psi.py --name="Qubit_psi" --dataset="all"
```
QubitE with $\psi = 0$ :
```shell
python train_QubitE_best.py --name="Qubit" --dataset="all"
```

