# oneNeuron
oneNeuron | perceptron

# commands used -
 
``` bash
 git add . && git commit -m "docstring updated" && git push origin main
 ``` 
## Shell commands
```
python
from utils.all_utils import prepare_data
```

'''
## Add URL -
[Git handbook] (https://guides.github.com/introduction/git-handbook/)

[README] (https://readme.so/editor)

<a href="https://www.google.com">visit google! </a>

## Add image - 
![sample Image] (plots/and.png)

<img src="plots/and.png" alt="Girl in a jacket" width="500" height="600">

## Python Code-left of 1 in keyboard(quote -`)

```python 
def main(data,eta,epochs):
  
  df=pd.DataFrame(data)
  print(df)

  X,y=prepare_data(df)
  model_AND =Perceptron(eta=ETA,epochs=EPOCHS)
  model_AND.fit(X,y)
  _=model_AND.total_loss()
  save_model(model_AND,filename="and.model")
  save_plot(df,"and.png",model_AND)
```

