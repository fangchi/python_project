# -*- coding: utf-8 -*
import pandas as pd
from math import log2
from pydotplus import graphviz

class Node(object):
    def __init__(self, attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init


# 生成决策树
def TreeGenerate(df):
    # 构建树的根节点
    new_node = Node(None, None, {})
    label_arr = df[df.columns[-1]]  # 获取数据集 DF 获取分类列表
    label_count = NodeLabel(label_arr)  # 将属性分类列表结果作为字典{分类 - 数量}
    if label_count:  # assert the label_count isn's empty
        new_node.label = max(label_count, key=label_count.get)  # 找到字典中结果分类数量最多的分类
        #  如果剩下的样本都是相同分类 或者 如果剩下的样本所有属性相同 结束构建
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node

        #  根据数据集获取信息熵和连续属性拆分值
        new_node.attr, div_value = OptAttr(df)

        # recursion
        if div_value == 0:  # categoric variable
            value_count = NodeLabel(df[new_node.attr])
            for value in value_count:
                df_v = df[df[new_node.attr].isin([value])]  # get sub set
                # delete current attribution
                df_v = df_v.drop(new_node.attr, 1)
                new_node.attr_down[value] = TreeGenerate(df_v)

        else:  # continuous variable # left and right child
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df[df[new_node.attr] <= div_value]  # get sub set
            df_v_r = df[df[new_node.attr] > div_value]

            new_node.attr_down[value_l] = TreeGenerate(df_v_l)
            new_node.attr_down[value_r] = TreeGenerate(df_v_r)

    return new_node


'''
make a predict based on root
@param root: Node, root Node of the decision tree
@param df_sample: dataframe, a sample line 
'''
def Predict(root, df_sample):
    try:
        import re  # using Regular Expression to get the number in string
    except ImportError:
        print("module re not found")

    while root.attr != None:
        # continuous variable
        if df_sample[root.attr].dtype == 'float' or df_sample[root.attr].dtype == 'int':
            # get the div_value from root.attr_down
            for key in list(root.attr_down):
                num = re.findall(r"\d+\.?\d*", key)
                div_value = float(num[0])
                break
            if df_sample[root.attr].values[0] <= div_value:
                key = "<=%.3f" % div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f" % div_value
                root = root.attr_down[key]

        # categoric variable
        else:
            key = df_sample[root.attr].values[0]
            # check whether the attr_value in the child branch
            if key in root.attr_down:
                root = root.attr_down[key]
            else:
                break

    return root.label


'''
calculating the appeared label and it's counts

@param label_arr: data array for class labels
@return label_count: dict, the appeared label and it's counts
'''


# 将List转化为中的值  按照{值类型-数量组成}字典
def NodeLabel(label_arr):
    label_count = {}  # store count of label
    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count



'''
 根据数据集DF获取信息熵 和分拆属性
 @:param df 数据集
 @:return opt_attr 裁剪属性(基于属性为离散)
 @:return div_value 拆分属性（基于属性为连续）
'''
def OptAttr(df):
    info_gain = 0
    for attr_id in df.columns[1:-1]:  # 遍历所有属性
        info_gian_tmp, div_value_tmp = InfoGain(df, attr_id)  # 获取对应属性的信息熵
        if info_gian_tmp > info_gain:
            info_gain = info_gian_tmp
            opt_attr = attr_id
            div_value = div_value_tmp

    return opt_attr, div_value

'''
根据数据集属性获取信息熵
@:param df 数据集
@:param index 信息熵
@:return info_gain 
'''
def InfoGain(df, index):
    info_gain = InfoEnt(df.values[:, -1])  # df.values[:, -1] 所有分类
    div_value = 0  # div_value for continuous attribute
    n = len(df[index])  # 获取数据集df对应属性的样本数量
    # 1.for continuous variable using method of bisection
    if df[index].dtype == 'float' or df[index].dtype == 'int': # 如果是数字类型 属性是连续类型
        sub_info_ent = {}  # store the div_value (div) and it's subset entropy
        df = df.sort_values([index], ascending=1)  # sorting via column
        df = df.reset_index(drop=True)
        data_arr = df[index]
        label_arr = df[df.columns[-1]]
        for i in range(n - 1):
            div = (data_arr[i] + data_arr[i + 1]) / 2
            sub_info_ent[div] = ((i + 1) * InfoEnt(label_arr[0:i + 1]) / n) \
                                + ((n - i - 1) * InfoEnt(label_arr[i + 1:-1]) / n)
        # our goal is to get the min subset entropy sum and it's divide value
        div_value, sub_info_ent_max = min(sub_info_ent.items(), key=lambda x: x[1])
        info_gain -= sub_info_ent_max
    else:  # 如果是字符类型
        data_arr = df[index]   # 获取属性值列表
        label_arr = df[df.columns[-1]]  # 获取总分类值
        value_count = NodeLabel(data_arr)  #获取属性-总数 字典

        for key in value_count: #遍历属性值
            key_label_arr = label_arr[data_arr == key]  #获取对应属性的的分类
            info_gain -= value_count[key] * InfoEnt(key_label_arr) / n

    return info_gain, div_value


'''
calculating the information entropy of an attribution

@param label_arr: ndarray, class label array of data_arr
@return ent: the information entropy of current attribution
'''


# 获取Array信息熵
def InfoEnt(label_arr):
    ent = 0
    n = len(label_arr)
    label_count = NodeLabel(label_arr)

    for key in label_count:
        ent -= (label_count[key] / n) * log2(label_count[key] / n)
    return ent


def DrawPNG(root, out_file):
    '''
    visualization of decision tree from root.
    @param root: Node, the root node for tree.
    @param out_file: str, name and path of output file
    '''
    g = graphviz.Dot()  # generation of new dot
    g.set_fontname('FangSong')
    TreeToGraph(0, g, root)
    g2 = graphviz.graph_from_dot_data(g.to_string())

    g2.write_png(out_file)


def TreeToGraph(i, g, root):
    '''
    build a graph from root on
    @param i: node number in this tree
    @param g: pydotplus.graphviz.Dot() object
    @param root: the root node

    @return i: node number after modified
#     @return g: pydotplus.graphviz.Dot() object after modified
    @return g_node: the current root node in graphviz
    '''
    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s" % (i, root.label, root.attr)
    g_node = i
    g.add_node(graphviz.Node(g_node, label=g_node_label,fontname="FangSong"))

    for value in list(root.attr_down):
        i, g_child = TreeToGraph(i + 1, g, root.attr_down[value])
        g.add_edge(graphviz.Edge(g_node, g_child, label=value,fontname="FangSong"))

    return i, g_node
