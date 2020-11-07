# BERT-Topic-Modeling

COVID19 BERT-Topic-Modeling is an NLP task meant to help identify hidden topics in a collection of documents.
In this work we employ Covid-19 Open Research Dataset and perform topic extraction on the first outbreak period between 2000 - 2020 year. The aformentioned dataset can be found at the following url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7251955/

Technologies used: Python3, Spacy, FastText, BERT, Jupyter Notebook, matplotlib

<p align="center"><img width="789" alt="Screenshot 2020-11-07 at 11 39 30" src="https://user-images.githubusercontent.com/11573356/98438928-0863de00-20ee-11eb-8527-078ec3890ef4.png"></p>

## Evaluation:

  |      Method   |     c_v       | u_mass    | silhouette | 
  | ------------- |:-------------:| ---------:|-----------:|
  |     **BERT**  | **0.524**     | -3.703    | -0.082     |
  
## Project Outline:
```
  - Generating ground-truth dataset
  - Model Training & Evaluation
  - Topic Prediction

Basic project installation steps:

  1. Clone repository

  2. Generate model & evaluation files:
     - preprocess list of documents
     - generate list of document tokens 
     - import and create Evaluation object
     - create model using create_model() function
     - save model & evaluation files to a given output path
     
     Sample:
          from evaluation import Evaluation
          
          token_lists = [
                          ["virus", "outbreak", "virus", "pandemic", "farm"], 
                          ["doctor", "risk_factor", "health", "health", "research"], 
                          ["outbreak", "hospital", "healthcare", "drug", "therapy", "illness"], 
                          ["death", "intervention", "factor", "people", "host", "transmission]
                        ]
          
          ev = Evaluation(lang_code="en", method="BERT", version="1.1", num_words=15)	
          ev.create_model(token_lists, output_path=output_path)	
    
     Evaluation files:
        - plot topic clusters 
        - topic wordclouds
        - evaluation metrics (c_v, u_mass, silhouette)
        - topic_terms 
  
  3. Predict topic for new documents:
      - import and create Topic object
      - predict topic using predict_topic() function

   Sample:
         from topic import Topic
         
         text = """ Kidney failure, also known as end-stage kidney disease, is a medical condition 
                    in which the kidneys are functioning at less than 15% of normal.
                """
         
         t = Topic(lang_code="en", method="BERT", version="1.1", k=134, num_words=5)
         pred = t.predict_topic(text)
         print(pred)
         
         '''    
               {
                 "topic_id":14,
                 "confidence":0.8932,
                 "topic_terms":[
                    {
                       "term":"kidney",
                       "weight":0.1733
                    },
                    {
                       "term":"nephrectomy",
                       "weight":0.0452
                    },
                    {
                       "term":"creatinine",
                       "weight":0.0269
                    },
                    {
                       "term":"injury",
                       "weight":0.0256
                    },
                    {
                       "term":"recipient",
                       "weight":0.0184
                    }
                 ],
                 "message":"successful"
              }
        '''
       
```
