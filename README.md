## Temporal Network Evolutionary Structure Learning using  Von Neumann Entropy Changes

### Requirements

Recent versions of PyTorch, numpy, sklearn, tqdm, joblib, and sacred. There are key required packages:
```python
python==3.7.0
joblib==1.1.0
numpy==1.22.2
pandas==1.4.1
scikit-learn==1.0.2
scipy==1.7.3
torch==1.4.0
sacred==0.8.2
tqdm==4.63.0
```

### Dataset format

The network should be in the following format:

- One line per interaction/edge.

- Each line should be:  *node u*, *node i*, *timestamp*, *label*,*edge index*.

- First line is the network format. 

- *Timestamp* should be in cardinal format (not in datetime).

-  If there are no labels, use 0 for all interactions.

- *Feature list* can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.

  For example, the first few lines of a dataset can be:

  ```python
  ,u,i,ts,label,idx
  0,3,403,12400.0,1.0,1
  1,11,971,12800.0,8.0,2
  2,11,272,12800.0,8.0,3
  3,114,55,12890.0,4.0,4
  ```

### Running the code

```python
  $ python experiments/experiment_learn_edges.py -g 0 -s bitcoinalpha -b 512 
```

If you want to change any settings, add a JSON file into  `./update_json/` or change the default settings from  `./ledge_configuration.py`.
