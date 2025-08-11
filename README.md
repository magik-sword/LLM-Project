![](UTA-DataScience-Logo.png)

# Using LLMs to Determine Poetry Authorship

This repository fine-tunes a GPT-2 model on a custom dataset of poems written by me and users from r/OCPoetry, then uses extracted features as inputs to a logistic regression model to determine whether a poem was written by me or by another author.

**Disclaimer:** The original dataset used for training and validation won't be uploaded here to comply with terms of service and for the privacy of the Reddit users. An example of the expected format for the data and instructions for creating it will be included in this repository.

## Overview

This project explores how well LLMs can generalize to new, unseen data in the form of original poetry I've written. A GPT-2 model with a classification head is fine-tuned on a binary labeled dataset of poems written by me and other users from r/OCPoetry. The embeddings of this trained model are then extracted and used as inputs to a simple logistic regression model that determines the authorship of each poem in the test set based on writing style. This approach is evaluated by comparing its performance to GPT-5 on the same test set using text analysis. The fine-tuned GPT-2 model embeddings combined with logistic regression achieved an accuracy of 83% compared to 50% by a baseline GPT-5 model.

## Summary of Work Done

### Data

* **Type:**
  * Input: CSV file containing 30 poems
  * Output: Text file containing predicted authorship next to each poem
* **Instances:**
  * Training: 24 poems
  * Testing: 6 poems

#### Data Collection and Preprocessing

* **Dataset Creation:** Collected poems from my personal notes and [r/OCPoetry](https://www.reddit.com/r/OCPoetry/). The CSV was manually created using the following guidelines:
  * Labeling each poem as 1 (written by me) or 0 (not written by me)    
  * Wrapping each poem in double quotes: ("The sky is blue.")
  * Representing linebreaks in poems with '\n'
  * Doubling the quotation marks if a poem begins with a quote ("Individuality" -> """Individuality"")
  * Doubling any internal quotes("He said, "Hello"!" -> "He said, ""Hello""!")
  * Breaking up some of my longer poems into shorter ones to increase the dataset size.
* **Preprocessing:** The following was done to the CSV file containing all the poems:
  * Shuffling the entries since all my 1s and 0s were grouped together
  * Converting the '\n' characters into actual line breaks
  * Splitting the dataset into 80/20 training and test sets
  
#### Data Visualization

GPT-2 Training vs Validation Loss 

![image](https://github.com/user-attachments/assets/126f0390-e17e-40b1-9501-eea13aad2ac1)

The validation loss started around 2.92 and decreased consistently all the way down to 1.85. The training loss decreased overall as well from 2.62 to 0.79, but jumped up at a few points such as epochs 3, 8, and 10. 
Despite the limited dataset, the model is still managing to learn from the data. 

Logistic Regression + Embeddings Confusion Matrix

![image](https://github.com/user-attachments/assets/f8465149-4b99-47b8-9d86-9e584980ff3e)

This was the result after scaling the embeddings and setting minimal hyperparameters (max_iter=1000, random_state=42) on the logistic regression model. The extracted features were highly effective as inputs for the model, with the model only misclassifying one poem.

#### Problem Formulation

  * **Models:**
    * GPT2ForSequenceClassification: Only last layer and classification head unfrozen
    * GPT-5: Accessed through a chat session on chatgpt.com
    * Logistic Regression: max_iter=1000, random_state=42
  * **Hyperparameters:**
    * learning_rate=5e-5
    * per_device_train_batch_size=2
    * per_device_eval_batch_size=2
    * num_train_epochs=10
    * weight_decay=0.01
    * seed=42
    
### Training

This project was done using my ASUS Zenbook Laptop. All the notebooks were run on Google Colab to use their NVIDIA T4 GPU to speed up training and loading in packages. Training both the LLM and logistic regression model only took a few minutes. A major challenge, especially in the beginning, was figuring out how many layers to unfreeze. Using only the classification head led to little learning and using the entire GPT-2 model quickly led to overfitting. A slightly higher learning rate also helped achieve the results shown in the repository.

### Performance Comparison

The dataset was created with an equal number of examples from both classes, so accuracy was the main metric used for evaluation. As mentioned in the overview, my approach achieved 83% accuracy compared to 50% for GPT-5 using only text analysis. When prompting GPT-5 in a chat session, I made sure to be logged out of my ChatGPT account to make sure it wouldn't pick up on my writing style tied to my account. The following shows the authorship predictions for each poem in the test set:

* Fine-Tuned GPT-2 Embeddings + Logistic Regression
  * Poem 1: Correct Prediction
  * Poem 2: Correct Prediction
  * Poem 3: Correct Prediction
  * Poem 4: Incorrect Prediction
  * Poem 5: Correct Prediction 
  * Poem 6: Correct Prediction
* ChatGPT-5 
  * Poem 1: Correct Prediction
  * Poem 2: Correct Prediction
  * Poem 3: Correct Prediction
  * Poem 4: Incorrect Prediction
  * Poem 5: Incorrect Prediction
  * Poem 6: Incorrect Prediction

### Conclusions

LLMs are highly effective at learning from new data and can be leveraged to capitalize on their strong generalization capabilities, especially with small, limited datasets.

### Future Work

* **Dataset Size:** Increasing the dataset size could likely lead to a smoother learning curve and allow for better classification.
* **Hyperparameters:** Experimenting with different values for the model hyperparameters, specifically batch size and learning rate, could help stabilize the training loss. 
* **Different Models:** Other OpenAI models could be used to see how well they perform on the dataset.
  
### How to Reproduce Results

* Create your own dataset using the guidelines and example.csv as reference 
* Download the GPT_2.ipynb notebook and open it in Google Colab
* Edit the notebook to your specific file path for the CSV
* Change runtime type to T4 GPU and run the notebook
* Paste the prompt.txt file into a ChatGPT-5 chat session and evaluate the results. 

### Overview of Files in Repository

* **example.csv:** Expected format for the dataset to use in training.
* **results.txt:** Authorship predictions alongside each poem in the test set.
* **prompt.txt:** Test set poems that can be given to ChatGPT-5 to make predictions on.
* **GPT_2.ipynb:** Preprocesses the data, trains, and evaluates the GPT-2 model. Creates prompt.txt and results.txt.

### Required Software

* Google Colab (Requires a Google account)
* VSCode, Excel, Google Sheets, Notepad, etc. (I used the text editor in VSCode to make my CSV file, but any of these programs will do.)
  
### Required Libraries

* Pandas
* PyTorch
* Transformers
* Datasets
* Matplotlib
* NumPy
* Scikit-Learn
* Tqdm
