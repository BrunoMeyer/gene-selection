# Gene Selection to Classification with Machine Learning Techniques
This project is an model to compare differents methods of feature selection in the context of microarray gene expression


This project uses the databases of GEMS© 2003-2005, Alexander Statnikov, Constantin F. Aliferis, Ioannis Tsamardinos, Discovery Systems Laboratory, Department of Biomedical Informatics, Vanderbilt University, Nashville, TN, USA 
The data is avaliable in: http://www.gems-system.org/


# Requirements
- python >= 3.65
  - scikit-learn >= 0.20
  - deap >= 1.2.2

# How to use
You can use the follow command to use the default databases:
```bash
./run.sh
```
This will create three log files with the results

Or, if you want to run on an specific database, try the follow command:

```bash
python3 main.py MY_DATABASE.txt 2>/dev/null
```

If you want to see the relevant genes selected, change the argument VIEW_SELECTED_GENES in the file `main.py`
