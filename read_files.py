import os
import re


def read_files(filetype):
    """
    filetype: 'train' or 'test'
    return:
    data: filetype数据集文本
    labels: filetype数据集标签
    """
    # 标签1表示正面，0表示负面
    labels = [1]*12500 + [0]*12500
    data = []
    file_list = []
    path = r'./aclImdb/'
    # 读取正面文本名
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path+file)
    # 读取负面文本名
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path+file)
    # 将所有文本内容加到all_texts
    for file_name in file_list:
        with open(file_name, encoding='utf-8') as f:
            data.append(rm_tags(" ".join(f.readlines())))
    return data, labels


#去除html标签
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)