import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
key_image=Image.open('dstree1.png')
key_image2=Image.open('dstree2.png')
st.set_page_config(
    layout='wide'
)
st.header('决策树模型')
with st.expander('模型释义'):
     st.image(key_image)
     st.image(key_image2)
st.subheader('决策树模型实践，以成年人收入分层为例子')
st.write('样例数据')
data = pd.read_csv('adult-change.csv')
st.dataframe(data.head())
st.write('请选择各项参数')
y = st.selectbox('自变量y', list(data.columns))
test_size = st.number_input('训练数据与测试数据分割比例')
if st.button('确认参数并开始训练模型'):
        x = data.drop([y], axis=1)
        y = data[y]
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100, test_size=test_size)
        # 数据标准化
        x_train = pd.get_dummies(x_train, dummy_na=True)
        x_test = pd.get_dummies(x_test, dummy_na=True)
        # 标准化后，出现特征值数量不一致情况
        x_test = x_test.drop(['native-country_Holand-Netherlands'], axis=1)
        # 模型实例化
        des = DecisionTreeClassifier()
        # 训练模型
        des.fit(x_train, y_train)
        # 预测
        y_pre = des.predict(x_test)
        # 模型评分
        score=accuracy_score(y_test,y_pre)
        st.write(f'模型得分：{score}')
