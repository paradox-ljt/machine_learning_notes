from sklearn import svm
import pandas as pd
import numpy as np
import sklearn

if __name__ == '__main__':
    file_path = r"C:/Users/华为/Desktop/程序/svm-data.xlsx"
    data = pd.DataFrame(pd.read_excel(file_path))
    print(r'''Info: 
    1. month: 1~12
    2. nation: 汉族
    3. sex：1(男), 0(女)
    4. Ispoorpot：0(Flase), 1(True)
    5. poor_level: 0(非贫困生), 1(家庭经济困难、家庭经济特别困难、突发事件特殊困难)
    6. dormi: 0(丁香), 1(海棠), 2(竹园)''')
    x, y = np.split(data.values, indices_or_sections=(6,), axis=1)
    x = x[:, 0:6]
    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
        x, y, random_state=1)
    classifier = svm.SVC(C=1, kernel='rbf', gamma=10,
                     decision_function_shape='ovo')  # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())
    print("训练集：", classifier.score(train_data, train_label))
    print("测试集：", classifier.score(test_data, test_label))
    a = classifier.decision_function(x)
    b = classifier.predict(x)
    print('train_decision_function:', a)
    print('predict_result:', b)
