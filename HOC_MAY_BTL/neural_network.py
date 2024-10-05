import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# Tải dữ liệu Auto MPG với tên cột đầy đủ
url = "https://www.kaggle.com/datasets/sadcougarx/cellphone-predictions/data"

# Giả sử bạn đã tải về và đọc nó như sau
data = pd.read_csv('HOC_MAY_BTL/SmartphonePrice.csv')

# Xử lý dữ liệu thiếu
data = data.dropna()
data = data[data['Price'] >= 0]


# Chọn thuộc tính đầu vào và biến mục tiêu
X = data[['Performance', 'Storage capacity', 'Camera quality', 'Battery life', 'Weight', 'age']]
y = data['Price']

# Chia dữ liệu thành tập train và test (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Từ tập train, tách thêm tập validation (15% của tập data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2142, random_state=42)

# Tiền xử lý dữ liệu: chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Xây dựng mô hình MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập train, valid và test
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)

# Tính toán lỗi
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
valid_mae = mean_absolute_error(y_valid, y_valid_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Tính toán R^2
train_r2 = r2_score(y_train, y_train_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Tạo DataFrame để theo dõi quá trình
results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [train_rmse, valid_rmse, test_rmse],
    'MAE': [train_mae, valid_mae, test_mae],
    'R^2': [train_r2, valid_r2, test_r2]
})

print("Kết quả lỗi, MAE và R^2:")
print(results)

# Vẽ biểu đồ dự đoán so với giá trị thực
plt.figure(figsize=(18, 6))

# Biểu đồ dự đoán so với giá trị thực cho dữ liệu train
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Giá trị thực (Train)')
plt.ylabel('Dự đoán (Train)')
plt.title('Dữ liệu Train')

# Biểu đồ dự đoán so với giá trị thực cho dữ liệu validation
plt.subplot(1, 3, 2)
plt.scatter(y_valid, y_valid_pred, alpha=0.5)
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='red', linestyle='--')
plt.xlabel('Giá trị thực (Validation)')
plt.ylabel('Dự đoán (Validation)')
plt.title('Dữ liệu Validation')

# Biểu đồ dự đoán so với giá trị thực cho dữ liệu test
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Giá trị thực (Test)')
plt.ylabel('Dự đoán (Test)')
plt.title('Dữ liệu Test')

plt.tight_layout()
plt.show()
import joblib
joblib.dump(model, 'HOC_MAY_BTL/neural_network_model.pkl')