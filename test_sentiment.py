import pandas as pd
import helper

data = {
    'user': ['Test', 'Test', 'Test', 'Test', 'Test'],
    'message': [
        "Mast party thi bro",
        "BC tu pagal hai",
        "Bohot badiya",
        "Bakwas hai sab",
        "Accha chal"
    ]
}

test_df = pd.DataFrame(data)

print("SVM Sentiment Test:")
print(helper.svm_sentiment("Overall", test_df))

print("\nLogistic Regression Sentiment Test:")
print(helper.ml_sentiment("Overall", test_df))
