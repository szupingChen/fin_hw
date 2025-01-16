import sys
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.model_selection import TimeSeriesSplit

news_data = pd.read_csv("C:\\Users\\User\\Downloads\\Combined_News_DJIA.csv")
djia_data = pd.read_csv("C:\\Users\\User\\Downloads\\upload_DJIA_table.csv")

# 添加 BERT 新聞處理函數
def process_news_bert(news_texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    features = []
    for _, row in news_texts.iterrows():
        # 合併該行的所有新聞文本
        combined_text = ' '.join(row.astype(str).values)
        inputs = tokenizer(combined_text, return_tensors='pt', 
                         max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 [CLS] token 的輸出作為整個序列的表示
            features.append(outputs.last_hidden_state[:, 0, :].numpy())
    
    return np.vstack(features)

# 添加技術指標計算函數
def add_technical_indicators(df):
    # 原有的技術指標
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 添加更多移動平均線
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    
    # EMA指標
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 添加更多動量指標
    df['ROC5'] = df['Close'].pct_change(periods=5) * 100
    df['ROC10'] = df['Close'].pct_change(periods=10) * 100
    df['ROC20'] = df['Close'].pct_change(periods=20) * 100
    df['MOM5'] = df['Close'].diff(periods=5)
    df['MOM10'] = df['Close'].diff(periods=10)
    df['MOM20'] = df['Close'].diff(periods=20)
    
    # 波動率指標
    df['STD10'] = df['Close'].rolling(window=10).std()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['STD60'] = df['Close'].rolling(window=60).std()
    
    # 添加ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # RSI with different periods
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['RSI9'] = calculate_rsi(df['Close'], 9)
    df['RSI25'] = calculate_rsi(df['Close'], 25)
    
    # MACD相關指標
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # 成交量指標
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    
    # 趨勢強度指標
    df['ADX'] = abs(df['MA20'] - df['MA60']) / df['STD20']
    df['Price_Level'] = (df['Close'] - df['MA20']) / df['STD20']
    df['Trend_Strength'] = abs(df['MA20'] - df['MA60']) / df['STD20']
    
    # 填充缺失值
    df.fillna(method='bfill', inplace=True)
    return df

# 修改 DJIA 數據處理函數
def process_djia(djia_data, window_size=10): 
    # 添加技術指標
    df = add_technical_indicators(djia_data)
    
    # 選擇特徵列
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'MA120',
        'EMA12', 'EMA26',
        'ROC5', 'ROC10', 'ROC20',
        'MOM5', 'MOM10', 'MOM20',
        'STD10', 'STD20', 'STD60',
        'ATR',
        'RSI14', 'RSI9', 'RSI25',
        'MACD', 'Signal_Line', 'MACD_Hist',
        'Volume_MA5', 'Volume_MA20', 'Volume_Ratio', 'OBV',
        'ADX', 'Price_Level', 'Trend_Strength'
    ]
    
    # 創建時間窗口特徵
    window_features = []
    
    # 為每個時間點創建包含過去window_size天數據的特徵
    for i in range(len(df)):
        if i < window_size:
            # 對於前window_size天，使用可用的所有數據
            past_data = df[feature_columns].iloc[:i+1].values
            # 用0填充不足的天數
            padding = np.zeros((window_size - past_data.shape[0], len(feature_columns)))
            window_data = np.vstack([padding, past_data])
        else:
            # 使用過去window_size天的數據
            window_data = df[feature_columns].iloc[i-window_size:i].values
        
        # 將窗口數據展平為一維數組
        window_features.append(window_data.flatten())
    
    # 轉換為numpy數組
    window_features = np.array(window_features)
    
    # 數據標準化
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(window_features)
    
    return torch.FloatTensor(scaled_features), scaler

# 簡化的特徵融合函數 - 直接串聯
def feature_fusion(news_features, market_features):
    # 確保兩個特徵集的樣本數量相同
    min_samples = min(len(news_features), len(market_features))
    news_features = news_features[:min_samples]
    market_features = market_features[:min_samples]
    
    # 直接串聯特徵
    fused_features = np.concatenate([news_features, market_features], axis=1)
    
    return fused_features

# 預測模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # L1正則化權重
        self.l1_lambda = 0.01
        hidden_dim = 64
        
        # 特徵提取層
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # 單層LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # 預測層
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1)
        )
        
        # 殘差連接
        self.shortcut = nn.Linear(input_dim, 1)
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 保存輸入用於殘差連接
        identity = self.shortcut(x)
        
        # 特徵提取
        features = self.feature_extractor(x)  # [batch_size, hidden_dim]
        
        # LSTM處理
        features = features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        lstm_out, _ = self.lstm(features)  # [batch_size, 1, hidden_dim]
        
        # 預測
        features = lstm_out.squeeze(1)  # [batch_size, hidden_dim]
        out = self.predictor(features)  # [batch_size, 1]
        
        # 添加殘差連接
        out = out + identity
        
        return out
    
    def get_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

def custom_loss(outputs, targets, model, alpha=0.6, beta=0.2, gamma=0.1):
    # 主要損失
    mse_loss = F.mse_loss(outputs, targets)
    huber_loss = F.huber_loss(outputs, targets, delta=1.0)
    
    # 方向損失
    if len(outputs) > 1:
        diff_pred = outputs[1:] - outputs[:-1]
        diff_target = targets[1:] - targets[:-1]
        direction_loss = -torch.mean(torch.sign(diff_pred) * torch.sign(diff_target))
    else:
        direction_loss = torch.tensor(0.0, device=outputs.device)
    
    # L1正則化損失
    l1_loss = model.get_l1_loss()
    
    return alpha * mse_loss + (1-alpha) * huber_loss + beta * direction_loss + gamma * l1_loss

# 主要處理流程
def main():
    # 修改訓練參數
    num_epochs = 150
    batch_size = 32
    window_size = 5
    
    # 1. 處理新聞文本
    news_features = process_news_bert(news_data.iloc[:, 2:])
    
    # 2. 處理 DJIA 數據，獲取特徵和scaler
    djia_features, price_scaler = process_djia(djia_data, window_size=window_size)
    
    # 保存日期資訊並按時間順序排序
    dates = pd.to_datetime(djia_data['Date'])
    
    # 按日期排序所有數據
    sort_idx = dates.argsort()
    dates = dates.iloc[sort_idx]
    news_features = news_features[sort_idx]
    djia_features = djia_features[sort_idx]
    
    # 3. 特徵融合（直接使用處理後的特徵）
    fused_features = feature_fusion(news_features, djia_features.numpy())
    
    # 4. 準備標籤（實際收盤價）
    labels = djia_data['Close'].values[sort_idx]
    
    # 數據標準化
    scaler = MinMaxScaler()
    labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))
    
    # 5. 分割訓練集和測試集（保持時間順序）
    train_size = int(0.8 * len(fused_features))
    
    X_train = fused_features[:train_size]
    X_test = fused_features[train_size:]
    y_train = labels_scaled[:train_size]
    y_test = labels_scaled[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]
    
    # 打印訓練集和測試集的日期範圍
    print(f"\n訓練集日期範圍: {train_dates.iloc[0].strftime('%Y-%m-%d')} 到 {train_dates.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"測試集日期範圍: {test_dates.iloc[0].strftime('%Y-%m-%d')} 到 {test_dates.iloc[-1].strftime('%Y-%m-%d')}")
    
    # 轉換為 PyTorch 張量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # 修改模型初始化時的輸入維度
    input_dim = fused_features.shape[1]  # 使用融合後的特徵維度
    predictor = StockPredictor(input_dim=input_dim)
    optimizer = torch.optim.AdamW(predictor.parameters(), 
                                lr=0.0003,
                                weight_decay=0.05,
                                betas=(0.9, 0.999))
    
    # 計算總步數
    steps_per_epoch = (len(X_train) + batch_size - 1) // batch_size
    
    # 修改學習率調度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # 修改早停機制
    patience = 20
    min_improvement = float('inf')
    patience_counter = 0
    best_model = None
    
    # 訓練循環
    for epoch in range(num_epochs):
        predictor.train()
        epoch_loss = 0
        num_batches = 0
        
        # 使用較大的批次進行訓練
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            batch_X = X_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            
            if len(batch_X) < 2:
                continue
            
            optimizer.zero_grad()
            outputs = predictor(batch_X).squeeze()
            loss = custom_loss(outputs, batch_y.squeeze(), predictor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # 驗證
        predictor.eval()
        with torch.no_grad():
            test_outputs = predictor(X_test).squeeze()
            test_outputs_original = scaler.inverse_transform(test_outputs.numpy().reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
            current_rmse = np.sqrt(mean_squared_error(y_test_original, test_outputs_original))
            
            # 早停檢查
            if current_rmse < min_improvement:
                min_improvement = current_rmse
                best_model = predictor.state_dict()
                patience_counter = 0  # 重置計數器
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 5 == 0:  # 更頻繁地打印訓練狀態
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_loss/num_batches:.6f}, '
                  f'RMSE: {current_rmse:.2f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # 加載最佳模型
    predictor.load_state_dict(best_model)
    
    # 8. 最終評估和可視化
    predictor.eval()
    with torch.no_grad():
        test_outputs = predictor(X_test).squeeze()
        
        # 轉換回原始價格範圍
        test_outputs_original = scaler.inverse_transform(test_outputs.numpy().reshape(-1, 1))
        y_test_original = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
        y_train_original = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
        
        # 打印訓練集最後5天的價格
        print("\n訓練集最後5天的價格:")
        for i in range(-5, 0):
            print(f"日期: {train_dates.iloc[i].strftime('%Y-%m-%d')}, 價格: {y_train_original[i][0]:.2f}")
        
        # 打印測試集的真實價格和預測價格
        print("\n測試集價格對比 (前5天和後5天):")
        print("\n前5天:")
        for i in range(5):
            print(f"日期: {test_dates.iloc[i].strftime('%Y-%m-%d')}")
            print(f"真實價格: {y_test_original[i][0]:.2f}")
            print(f"預測價格: {test_outputs_original[i][0]:.2f}")
            print(f"誤差: {abs(y_test_original[i][0] - test_outputs_original[i][0]):.2f}")
            print("---")
        
        print("\n後5天:")
        for i in range(-5, 0):
            print(f"日期: {test_dates.iloc[i].strftime('%Y-%m-%d')}")
            print(f"真實價格: {y_test_original[i][0]:.2f}")
            print(f"預測價格: {test_outputs_original[i][0]:.2f}")
            print(f"誤差: {abs(y_test_original[i][0] - test_outputs_original[i][0]):.2f}")
            print("---")
        
        # 計算所有評估指標
        rmse = np.sqrt(mean_squared_error(y_test_original, test_outputs_original))
        mae = np.mean(np.abs(test_outputs_original - y_test_original))
        mape = np.mean(np.abs((y_test_original - test_outputs_original) / y_test_original)) * 100
        
        print(f"\n均方根誤差 (RMSE): {rmse:.2f}")
        print(f"平均絕對誤差 (MAE): {mae:.2f}")
        print(f"平均絕對百分比誤差 (MAPE): {mape:.2f}%")
        
        # 設置繪圖風格和中文字型
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建單一圖表
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 繪製預測結果對比圖
        ax.plot(test_dates, y_test_original, label='實際價格', color='blue', linewidth=2)
        ax.plot(test_dates, test_outputs_original, label='預測價格', color='red', linestyle='--', linewidth=2)
        ax.set_title('道瓊斯指數預測結果對比', fontsize=14, pad=20)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('價格', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 旋轉日期標籤以避免重疊
        ax.tick_params(axis='x', rotation=45)
        
        # 設置日期格式化
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 添加評估指標文字
        metrics_text = f'評估指標:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%'
        fig.text(0.15, 0.95, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # 調整圖表佈局
        plt.tight_layout()
        
        # 保存高質量圖片到指定路徑
        save_path = "C:\\Users\\User\\Desktop\\stock_prediction_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n圖表已保存至: {save_path}")
        plt.close()
    
    return predictor, rmse

if __name__ == "__main__":
    model, rmse = main()
    print(f"\n模型訓練完成，最終 RMSE: {rmse:.2f}")