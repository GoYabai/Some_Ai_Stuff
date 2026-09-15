import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. TẠO DỮ LIỆU GIẢ LẬP (1000 trận đấu của Bot)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Search_Depth': np.random.randint(3, 10, n_samples),
    'Center_Control': np.random.uniform(0.1, 1.0, n_samples),
    'Opening_Book': np.random.choice([0, 1], n_samples),
    'Time_Limit': np.random.randint(500, 3000, n_samples)
})

# Điều kiện thắng giả lập: Sâu > 5, kiểm soát trung tâm > 0.5 và có sách khai cuộc
data['Win'] = np.where(
    (data['Search_Depth'] > 5) & 
    (data['Center_Control'] > 0.5) & 
    (data['Opening_Book'] == 1), 
    1, 0
)

# Thêm chút nhiễu (noise) để cây quyết định không bị hoàn hảo 100%
noise_indices = np.random.choice(data.index, size=50, replace=False)
data.loc[noise_indices, 'Win'] = 1 - data.loc[noise_indices, 'Win']

X = data.drop('Win', axis=1)
y = data['Win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# PHẦN THỰC HÀNH CỦA ANH BẮT ĐẦU TỪ ĐÂY
# ==========================================

# TODO 1: Khởi tạo và huấn luyện mô hình DecisionTreeClassifier (max_depth=3)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# CHỌN 1 TRẬN ĐẤU ĐỂ GIẢI THÍCH (Ví dụ: trận thứ 42 trong tập test)
SAMPLE_ID = 42
sample_x = X_test.iloc[[SAMPLE_ID]]

# TODO 2: Khởi tạo SHAP Explainer
# Gợi ý: Với Decision Tree, thuật toán tối ưu nhất là TreeExplainer.
# Hãy truyền mô hình của anh vào hàm shap.TreeExplainer()
explainer = shap.TreeExplainer(model)

# TODO 3: Tính toán SHAP values cho sample_x
shap_values = explainer(sample_x)

# TODO 4: Vẽ biểu đồ Waterfall để giải thích kết quả
# Lệnh vẽ biểu đồ waterfall (Lưu ý: Đối với bài toán phân loại phân nhánh (0/1), 
# hãy truyền vào shap_values[0][:, 1] để lấy biểu đồ cho class '1' - Win)
shap.waterfall_plot(shap_values[0][:, 1])

plt.show()