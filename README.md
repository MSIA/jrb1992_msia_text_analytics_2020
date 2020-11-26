# jrb1992_msia_text_analytics_2020
**Model Training**

For model training via logistic regression, SVM, Fasttext, and CNN, open terminal and run command:
```
python Model_training.py
```
The best fasttext model is saved into 'best_fasttext.bin'. Due to the large file size of 807MB, it is uploaded to google drive folder "Project_Data".
Please access the model via link: 

For running the Flask App, please open two terminal windows, and run these commands separately:
```
python api.py
```
```
python client.py
```
By replacing text in json file in  "client.py", you can type in reviews and generate predicted label and probability.
For example, if you type this review:
```
response = requests.get("http://127.0.0.1:5000/predict", json={"text":"This tastes very sweet."})
```
It will return response like below:
```
<Response [200]>
{'Label': "('__label__Chardonnay',)", 'Probability': '[0.91913831]', 'Review': 'This tastes very sweet.'}
```
