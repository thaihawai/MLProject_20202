# MLProject_20202
## IMPORTANT
- The project is run entirely on Google Colab, therefore in each notebook the first cell (containing "import os" and "os.chdir...") is used to move into the project directory on Google Drive.
- When running on local machine this cell can be omitted.
## Job Postings Prediction
- The dataset is found on Kaggle: kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
- The objective is to train a model that can detect whether a job postings is fake or not
- After preprocessing, the input consists of two parts: textual data and categorical data
- In order to transform input data into vectors so that we can train the model, the approach is to try to summarize textual data into context vectors of float numbers
- Two methods are used in order to perform this embedding:
- The first is a LSTM/BiLSTM model that is trained on the text data, using job titles as output variable. This model is trained to summarize the text
- The second is by using Doc2Vec on the whole text data (no need to split into input/output since Doc2Vec does not need an output)
## Order of notebook to run
- Within the folder Source_Code, each notebook is labeled with a number denoting their running sequence .i.e 01->02->03...
- There are two embedding methods and they can be run independently
- The embedding model using LSTM is achieved by running 04a->04.5a, the embedding model using Doc2Vec is in 04b
- Afterward run by sequence
