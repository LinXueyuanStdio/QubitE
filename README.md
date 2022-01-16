# QubitE

code for "QubitE: Qubit Embedding for Knowledge Graph Completion".

## Environment

- PyTorch 1.8.1 + cuda 10.2

## How to Run

QubitE with $\psi != 0$ :
```shell
python train_QubitE_best_psi.py --name="QubitE_psi"
```
QubitE with $\psi = 0$ :
```shell
python train_QubitE_best.py --name="QubitE"
```
