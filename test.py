import numpy as np

# 创建两个二维数组
array1 = np.array([[1, 2, 3,5],
                   [4, 5, 6,9]])

array2 = np.array([[7, 8, 9,13],
                   [10, 11, 12,15]])

# 使用 np.stack 沿着新的轴（axis=2）堆叠数组
stacked_array = np.stack((array1, array2), axis=2)

print("Original arrays:")
print(array1)
print(array2)
print("\nStacked array along axis=2:")
print(stacked_array)
