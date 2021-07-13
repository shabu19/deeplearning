import os
import pandas as pd
import xlsxwriter

d = pd.read_csv('C:/Users/User/Desktop/data_list.csv', header=None, usecols=[0,1,2,3])

dl_col_0 = d[0]
dl_col_1 = d[1]
dl_col_2 = d[2]
dl_col_3 = d[3]

classes = []
y = 0
tt_list = []
new_tt_list = []

for x in dl_col_1:
    if x in classes:
        classes = classes
    else:
        classes.append(x)

for x in range(101):
    tt_list = []
    for y in range(13320):
        if classes[x] is dl_col_1[y]:
            tt_list.append(dl_col_0[y])
    for z in range(len(tt_list)):
        l = len(tt_list) - 1
        if z <= l * 5 / 100:
            new_tt_list.append('test')
        else:
            new_tt_list.append('train')

workbook = xlsxwriter.Workbook('train_test_split.csv')
worksheet1 = workbook.add_worksheet()

worksheet1.write_column('A1', new_tt_list)

workbook.close()
