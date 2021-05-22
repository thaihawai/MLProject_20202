# MLProject_20202
## Job Postings Prediction
### The dataset is found on Kaggle: kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
### The objective is to train a model that can detect whether a job postings is fake or not
### After preprocessing, the input consists of two parts: textual data and categorical data
### In order to transform input data into vectors so that we can train the model, the approach is to try to summarize textual data into context vectors of float numbers
### Two methods are used in order to perform this embedding:
### The first is a LSTM/BiLSTM model that is trained on the text data, using job titles as output variable. This model is trained to summarize the text
### The second is by using Doc2Vec on the whole text data (no need to split into input/output since Doc2Vec does not need an output)
## Order of notebook to run
### Within the folder Source_Code, each notebook is labeled with a number denoting their running sequence .i.e 01->02->03...
### For notebooks with letter 'a' or 'b' after their number, they are two 'paths' or the process corresponding to the two embedding method
### Users can choose either 'path' to run or both .i.e 03.5a->04a->05a->...; 04b->05b->06b->...;
### The two 'path' are independent and won't interfere with each other
