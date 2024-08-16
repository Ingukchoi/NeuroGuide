# **NeuroGuide: Learn to Collaborate for the Clustered Traveling Salesman Problem**

### **Dependencies:**
---
- `Python=3.10.14`
- `torch==2.2.2`
- `torch_geometric==2.5.2`
- `numpy==1.24.3`
- `pytz==2024.1`
- `sklearn==1.4.2`

### **Training the model:**
---
- Run `train_N_M.py`. You can adjust parameters in `train_N_M.py`. It's currently set to train for `N=20, M=3` and `N=50, M=5` problems. The hyperparameters are configured as described in the paper, maintaining consistency with the original research.

### **Testing the model:**
---
- Run `test_N.py`. Choose the model to test by modifying the parameter in `test_N.py` (seed: 1235). It's set to use the trained model (`N=20, M=3` and `N=50, M=5`) in the result folder, but you can switch to a model you've trained.

### **Acknowledgements:**
---
- NeuroGuide's code execution is based on the [POMO](https://github.com/yd-kwon/POMO). We thank them for their contribution.
