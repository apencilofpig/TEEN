# 原始字典
original_map = {    0: 0, 1: 16, 2: 17, 3: 18, 4: 19, 5: 20, 6: 21, 7: 22, 
    8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
    16: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6, 
    22: 7, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 
    28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 
    34: 34, 35: 35
}

# 1. 一行代码完成转换和排序，得到新的字典
sorted_swapped_map = {v: k for k, v in sorted(original_map.items(), key=lambda item: item[1])}


# 2. 遍历新的字典，并按 "key: val" 格式换行输出
print("--- 格式化输出 ---")
for key, val in sorted_swapped_map.items():
  print(f"{key}: {val}")