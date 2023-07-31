The Repo Structure
-------------------
- The repo contains 6 main folders.
- 1 folder named "my_package" which contains classes that were created and are imported in the notebooks.
- These classes are required for data cleaning, preprocessing, and modeling functionalities.
- The repo also has 5 other folders, one folder for each step of the project being named "step x" where "x" denotes the step number.
- Each step folder will contain jupyter notebooks of the experiments that were carried out during this step.
	- step 0: EDA and Datasets Comparison. Please Notice that EDA notebook is added as a compressed html page due to the file size.
	- step 1: Hyperparameters Tuning Notebooks.
	- step 2: Preprocessing Pipelines Experiments Notebooks.
	- step 3: Class Imbalance SOTA techniques Experiments Notebooks.
	- step 4: Applying the best approach on all the 6 dataset variants.
- We didn't prefer to write everything in one notebook such that the code and experiments become more organized.
- The repo also contains the "requirements.txt" file that contains the project dependencies.
- We also added the project presentation in pptx and pdf formats.

Please note that the repo contains a folder named "Examples" but it can be ignored.


How to Reproduce the Results
-----------------------------
1- First please go to the following Kaggle link to download the data CSV files: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Base.csv
2- Import the downloaded datasets into the notebook.
3- Use the "requirements.txt" file to install the project libraries dependencies using pip install -r "requirements.txt"
4- Restart your notebook kernel for updates to takeplace.
5- Import all the classes that are in "my_package" folder to the notebook.
6- After doing the above steps the notebooks will run without errors.





	
