names = ["小明", "小红", "小刚"]
scores = [85, 92, 78]

# 使用 zip 把它们缝合起来
combined = zip(names, scores)

# 转换成列表看一眼结果
print(list(combined))
# 输出: [('小明', 85), ('小红', 92), ('小刚', 78)]
for name, score in zip(names, scores):
    print(f"{name} 考了 {score} 分")