import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Đọc file CSV
data = pd.read_csv('HOC_MAY_BTL/SmartphonePrice.csv')

# Tách dữ liệu thành X (features) và y (target)
X = data.drop(columns=['Price'])
y = data['Price']

# Chuẩn hóa các giá trị trong X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Giảm chiều dữ liệu với PCA, giữ lại 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Ghi lại file CSV với dữ liệu đã chuẩn hóa và giảm chiều
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# Kết hợp cả tập train và test
X_scaled_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
X_scaled_df['Price'] = pd.concat([y_train, y_test], ignore_index=True)

# Ghi ra file CSV mới
X_scaled_df.to_csv('HOC_MAY_BTL/SmartphonePrice_scaled_pca.csv', index=False)

print("File has been saved as 'SmartphonePrice_scaled_pca.csv'")
