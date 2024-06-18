from libsvm.svmutil import *
import pickle
from icecream import ic    

def read_pkl(path):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    # convert labels
    df['diagnosis'] = df['diagnosis'].map({'M': -1., 'B': 1.})
    y = df['diagnosis'].tolist()
    
    # convert data
    data_cols = df.columns[2:]
    x = []
    index_counter = 1
    for index, row in df.iterrows():
        row_dict = {}
        for col in data_cols:
            row_dict[index_counter] = row[col]
            index_counter += 1
        # reset index_counter
        index_counter = 1
        x.append(row_dict)
    assert len(y) == len(x), 'Length of y and x must be the same'
    return y, x      
    

if __name__ == '__main__':
    y, x = read_pkl('data.pkl')
    train_y, train_x = y[:400], x[:400]
    test_y, test_x = y[400:], x[400:]
    # define problem and parameter
    prob = svm_problem(train_y, train_x)
    param = svm_parameter('-t 0 -c 8.0 ') 
    model = svm_train(prob, param)
    svm_save_model('model.model', model)
    p_label, p_acc, p_val = svm_predict(train_y, train_x, model)
    ic(p_acc)
    p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
    ic(p_acc)