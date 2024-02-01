from sklearn import tree
import graphviz
import pandas as pd
import numpy as np

file_path = r"C:/Users/华为/Desktop/程序/svm-data.xlsx"
data = pd.DataFrame(pd.read_excel(file_path))
print(r'''Info: 
1. month: 1~12
2. nation: 汉族
3. sex：1(男), 0(女)
4. Ispoorpot：0(Flase), 1(True)
5. poor_level: 0(非贫困生), 1(突发事件特殊困难), 2(家庭经济困难), 3(家庭经济特别困难)
6. dormi: 0(丁香), 1(海棠), 2(竹园)''')
x, y = np.split(data.values, indices_or_sections=(6,), axis=1)
#绘制树模型
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
print(clf.predict(x))
tree.export_graphviz(clf)
dot_data = tree.export_graphviz(clf, out_file=None)
graphviz.Source(dot_data)
#利用render方法生成图形
graph = graphviz.Source(dot_data)
graph.render("tree")