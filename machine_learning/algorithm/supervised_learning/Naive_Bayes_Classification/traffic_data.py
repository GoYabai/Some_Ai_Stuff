import numpy as np


# ==========================================
# PHẦN 1: DỮ LIỆU ĐÃ CHUẨN BỊ SẴN
# ==========================================
def create_traffic_data():
    """Tập dữ liệu Traffic Data (20 mẫu)"""
    data = [
        ['Weekday', 'Spring', 'None',   'None',   'On Time'],
        ['Weekday', 'Winter', 'None',   'Slight', 'On Time'],
        ['Weekday', 'Winter', 'None',   'None',   'On Time'],
        ['Holiday', 'Winter', 'High',   'Slight', 'Late'],
        ['Saturday','Summer', 'Normal', 'None',   'On Time'],
        ['Weekday', 'Autumn', 'Normal', 'None',   'Very Late'],
        ['Holiday', 'Summer', 'High',   'Slight', 'On Time'],
        ['Sunday',  'Summer', 'Normal', 'None',   'On Time'],
        ['Weekday', 'Winter', 'High',   'Heavy',  'Very Late'],
        ['Weekday', 'Summer', 'None',   'Slight', 'On Time'],
        ['Saturday','Spring', 'High',   'Heavy',  'Cancelled'],
        ['Weekday', 'Summer', 'High',   'Slight', 'On Time'],
        ['Weekday', 'Winter', 'Normal', 'None',   'Late'],
        ['Weekday', 'Summer', 'High',   'None',   'On Time'],
        ['Weekday', 'Winter', 'Normal', 'Heavy',  'Very Late'],
        ['Saturday','Autumn', 'High',   'Slight', 'On Time'],
        ['Weekday', 'Autumn', 'None',   'Heavy',  'On Time'],
        ['Holiday', 'Spring', 'Normal', 'Slight', 'On Time'],
        ['Weekday', 'Spring', 'Normal', 'None',   'On Time'],
        ['Weekday', 'Spring', 'Normal', 'Heavy',  'On Time']
    ]
    return np.array(data)

def get_feature_index(feature_value, feature_values_array):
    """Hàm helper để tìm vị trí index của một giá trị đặc trưng"""
    indices = np.where(feature_values_array == feature_value)[0]
    return indices[0] if len(indices) > 0 else -1


# ==========================================
# PHẦN 2: BÀI TẬP THỰC HÀNH
# ==========================================

def compute_prior_log_probabilities(train_data):
    """
    Yêu cầu: Tính log(P(c)) cho tất cả các lớp c
    """
    class_names = np.unique(train_data[:, -1])
    total_samples = len(train_data)
    prior_log_probs = np.zeros(len(class_names))

    for i, class_name in enumerate(class_names):
        # 1. Đếm số lượng mẫu thuộc lớp class_name
        # 2. Tính tỷ lệ P(c)
        # 3. Tính log tự nhiên (np.log) của P(c) và gán vào prior_log_probs[i]
        
        ### YOUR CODE HERE ###
        class_count = np.sum(train_data[:, -1] == class_name)
        p = class_count / total_samples
        prior_log_probs[i] = np.log(p)
        
    return prior_log_probs, class_names

def compute_conditional_log_probabilities(train_data, class_names, alpha=1.0):
    """
    Yêu cầu: Tính log(P(x_i|c)) có áp dụng Laplace Smoothing
    """
    n_features = train_data.shape[1] - 1
    conditional_log_probs = []
    feature_values = []

    for feature_idx in range(n_features):
        unique_values = np.unique(train_data[:, feature_idx])
        feature_values.append(unique_values)
        num_unique_values = len(unique_values) # Chính là |V_i| trong công thức
        feature_cond_log = np.zeros((len(class_names), len(unique_values)))

        for class_idx, class_name in enumerate(class_names):
            mask = train_data[:, -1] == class_name
            sample = train_data[mask]
            sample_size = len(sample)

            for value_idx, value in enumerate(unique_values):
                # 1. Đếm số lần 'value' xuất hiện trong 'sample' (lưu vào biến feature_count)
                # 2. Tính xác suất smoothed_prob = (feature_count + alpha) / (sample_size + alpha * num_unique_values)
                # 3. Tính log của smoothed_prob và gán vào feature_cond_log[class_idx][value_idx]
                
                ### YOUR CODE HERE ###
                feature_count = np.sum(sample[:, feature_idx] == value)
                smoothed_prob = (feature_count + alpha) / (sample_size + alpha * num_unique_values)
                feature_cond_log[class_idx][value_idx] = np.log(smoothed_prob)
                
        conditional_log_probs.append(feature_cond_log)

    return conditional_log_probs, feature_values

def predict_traffic(X, prior_log_probs, conditional_log_probs, feature_values, class_names):
    """
    Yêu cầu: Dự đoán nhãn cho mẫu X mới
    """
    log_scores = np.zeros(len(class_names))

    for class_idx in range(len(class_names)):
        # 1. Khởi tạo điểm số ban đầu bằng log xác suất tiên nghiệm của lớp này
        score = prior_log_probs[class_idx]
        
        for feature_idx, feature_value in enumerate(X):
            value_idx = get_feature_index(feature_value, feature_values[feature_idx])
            
            # 2. Nếu value_idx != -1 (tức là giá trị có tồn tại), 
            # hãy cộng dồn log xác suất có điều kiện tương ứng vào 'score'
            
            ### YOUR CODE HERE ###
            if value_idx != -1:
                score += conditional_log_probs[feature_idx][class_idx, value_idx]
            
        log_scores[class_idx] = score

    # Phần này em để sẵn kỹ thuật Log-sum-exp trick để tham khảo
    # xử lý tràn số (overflow/underflow) khi chuyển từ log về lại xác suất phần trăm
    log_scores_shifted = log_scores - np.max(log_scores)
    probabilities = np.exp(log_scores_shifted)
    normalized_probs = probabilities / np.sum(probabilities)

    predicted_class_idx = np.argmax(normalized_probs)
    prediction = class_names[predicted_class_idx]

    return prediction


# ==========================================
# PHẦN 3: TEST CASE ĐỂ KIỂM TRA
# ==========================================
if __name__ == "__main__":
    train_data = create_traffic_data()
    prior_log, class_names = compute_prior_log_probabilities(train_data)
    cond_log, feat_vals = compute_conditional_log_probabilities(train_data, class_names, alpha=1.0)
    
    X_test = ['Holiday', 'Winter', 'High', 'Heavy']
    prediction = predict_traffic(X_test, prior_log, cond_log, feat_vals, class_names)
    print(f"Dữ liệu đầu vào: {X_test}")
    print(f"Dự đoán của bạn: {prediction}")
    # Kết quả kỳ vọng: Nếu code chuẩn, hàm sẽ in ra 'Very Late' hoặc 'Late' tùy vào độ mượt
