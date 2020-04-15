# MC-Assignment-2
CSE 535: Mobile Computing Assignment-2 (Gesture Recognition + API Hosting)

This assignment is to classify the keypoints data as the type of gesture from one among ['communicate', 'hope', 'mother', 'buy', 'really', 'fun].
The training data for classification is given and we trained our model using LogisticRegression,
Decision Tree, MLP and Random forest. We train a separate models to do a binary classification for each gesture.
During prediction, we extract gesture specific features and feed it to each of the binary classifiers.
We assign the gesture whose model gives the highest probability score. 

## Testing the assignment

  1. Place the test files in the JSON_Test Folder under the appropriate folders according to the gesture names
  2. Make sure to run the notebook/feature_vector_* files by uncommenting the modeling_* function call inside each of 
  these files in order to populate the models folder with the pickle files
  3. Run the server.py file to start serving at localhost
  4. As the model is already trained and stored as a pickle file, Please navigate
     inside the Code folder and run the `client.py`
     ```bash
        > cd notebook
        > python3 client.py
     ```
  5. The client.py file displays the results obtained from the server after processing 
  the files in Test Folder

    +-- JSON_Test
    |   +-- buy
    |   |  +-- Place your test file if it is buy
    |   +-- hope
    |   |  +--- Place your test file here if it is hope
    +-- server.py - Run this
    +-- notebook
    |   +-- feature_vector_* - Run to populate models folder
    |   +-- client.py - Then Run this
  ```
    Note: The results will be displayed in client.py if everything works well
  ```
## Files and it's usage

`utils.py` - Contains common functions shared across all files like feature_matrix_extraction, 
                important dictionaries which hold GESTURE_INDEX and MODEL_INDEX, etc.

`feature_vector_*.py` - Contains code to extract the feature vector for the specified 
                        gesture. Can also be used to populate models folder during training

`client.py` - Queries the server with the JSON_Test data and evaluates the models

`pca_reduction.py` -  Reduces the data dimensions of data to preserve 95% variance after recognizing
                    largely varying features and transforming them.

`server.py` -  Hosts the API at a given IP Address and receives requests, processes them and 
                sends back responses
