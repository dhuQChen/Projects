import csv
import os
import pandas as pd

# 读取所有表中的职业（pros.xlsx中有所有职业）
pros_path = 'professions/pros.xlsx'
df = pd.read_excel(pros_path, header=None)
# 获取第一列病转化为列表
pros_list = df[0].values.tolist()
print(pros_list)

# 原始数据根目录
root_path = "data"
dirnames = os.listdir(root_path)


for k in range(len(pros_list)):
    print('--------')
    print(pros_list[k])

    # 职业名字中可能含有字符'/'，会影响路径查找，所以将职业名字中的'/'替换为'and'
    csv_path = "DB/" + eval(repr(pros_list[k]).replace('/', 'and')) + '.csv'
    print(csv_path)

    # 打开文件，如果不存在该文件将创建
    with open(csv_path, 'w+') as f:
        csv_write = csv.writer(f, lineterminator='\n')

        for dir in dirnames:
            filenames = os.listdir(root_path + os.sep + dir)
            # print(filenames)
            for file in filenames:
                # all excel
                data = pd.read_excel(root_path + os.sep + dir + os.sep + file, header=None)

                # 从第四行开始为有效数据
                data = data[3:][1:]

                # 由于后续要根据职业名称查找，所以此处需要将第一列作为索引
                data.set_index(1)
                csv_columns = [dir + os.sep + file.strip().split('.')[0]]


                # Data
                previous = 0.0
                row_list = data[data[1] == pros_list[k]].values.tolist()
                if len(row_list) != 0:
                    # previous = row_list[0][1]
                    for i in range(2, len(row_list[0])):
                        csv_columns.append(row_list[0][i])

                # Have no demand
                # 五匹配数据，使用0填充
                else:
                    for i in range(10):
                        csv_columns.append(previous)
                csv_write.writerow(csv_columns)
    f.close()


