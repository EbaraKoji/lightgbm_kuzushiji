from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # noqa F401
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score

file_path = '../datasets/kuzushiji_MNIST'
classes_file_path = f'{file_path}/kmnist_classmap.csv'
trainX_file_path = f'{file_path}/kmnist-train-imgs.npz'
trainY_file_path = f'{file_path}/kmnist-train-labels.npz'
testX_file_path = f'{file_path}/kmnist-test-imgs.npz'
testY_file_path = f'{file_path}/kmnist-test-labels.npz'

classes = pd.read_csv(classes_file_path)
# print(classes.shape)  # (10, 3)
# print(classes)
#    index codepoint char
# 0      0    U+304A    お
# 1      1    U+304D    き
# ...
label_index = classes.set_index('index').to_dict()['char']

npz_key_name = 'arr_0'
x_train: np.ndarray = np.load(trainX_file_path)[npz_key_name]
y_train: np.ndarray = np.load(trainY_file_path)[npz_key_name]
x_test: np.ndarray = np.load(testX_file_path)[npz_key_name]
y_test: np.ndarray = np.load(testY_file_path)[npz_key_name]

# print(x_train.shape)  # (60000, 28, 28)
# print(x_test.shape)  # (10000, 28, 28)
# print(y_train[0:5])  # [8 7 0 1 4]
# print(x_train.min(), x_train.max())  # 0 255

# print(label_index[y_train[0]])  # れ
# plt.imshow(x_train[0], cmap=plt.cm.gray)
# plt.show()

# preprocessing
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
# print(x_train[0, 10:15])  # [0.91372549 0.3372549  0.         0.         0.1254902 ]

lgb.LGBMClassifier(random_state=0)

train_data = lgb.Dataset(x_train, label=y_train)
eval_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 10,
    'verbose': 0,
}

gbm = lgb.train(
    params,
    train_data,
    valid_sets=eval_data,
    num_boost_round=100,
    verbose_eval=5,
)

preds = gbm.predict(x_test)
y_pred: List[int] = []
for x in preds:
    y_pred.append(np.argmax(x))

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# [[900   2   4   4  28   8   5  28  19   2]
#  [  1 869  26   2  16   6  41   7  15  17]
#  [ 10  15 852  48  12   5  23  11  14  10]
#  [  2   9  30 936   6   3   6   3   1   4]
#  [ 25  15  24   9 863   6  26   8  14  10]
#  [  6  20  77   6  14 835  30   2   6   4]
#  [  2   9  37   5  15   6 916   6   2   2]
#  [  8   8  16   3  49   1  35 830  33  17]
#  [  3  25  10  26   0   6  15   3 907   5]
#  [  6  21  21   6  35   0   8  17  17 869]]
# 0.8777
