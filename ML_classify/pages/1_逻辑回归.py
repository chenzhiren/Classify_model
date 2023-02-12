import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
key_image=Image.open('logit.png')
st.set_page_config(
    layout='wide'
)
st.header('逻辑回归模型')
with st.expander('模型释义'):
     st.image(key_image)
st.subheader('逻辑回归实践，以商品分类数据为例子')
st.write('样例数据')
data = pd.read_csv('Otto_train.csv')
st.dataframe(data.head())
st.write('请选择各项参数')
y = st.selectbox('自变量y', list(data.columns))
penalty=st.selectbox('惩罚函数(可选l1,l2)',['l1','l2','None'])
c=st.number_input('超参数')
class_weight=st.selectbox('各分类样本权重比例',['balanced','dict','None'])
solver=st.selectbox('优化求解算法',['newton-cg','liblinear','sag','saga','lbfgs','None'])
n_jobs=st.number_input('调用运行芯片数')
test_size = st.number_input('训练数据与测试数据分割比例')
if st.button('确认参数并开始训练模型'):
        data=data.set_index('id')
        x = data.drop([y], axis=1)
        y = data[y]
        y = y.apply(lambda x: x[6:])
        y = y.apply(lambda x: int(x) - 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=100)
        # 模型实例化
        logist = LogisticRegression(n_jobs=int(n_jobs),penalty=penalty,C=c,class_weight=class_weight,solver=solver)
        # 训练模型
        logist.fit(x_train, y_train)
        # 预测
        y_pre = logist.predict(x_test)
        # 模型评分
        score = accuracy_score(y_test, y_pre)
        st.write(f'模型得分：{score}')