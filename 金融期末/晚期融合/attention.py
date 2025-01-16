import sys
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, pipeline
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
import math

# 在文件開頭添加設備檢查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

news_data = pd.read_csv("C:\\Users\\User\\Downloads\\Combined_News_DJIA.csv")
djia_data = pd.read_csv("C:\\Users\\User\\Downloads\\upload_DJIA_table.csv")

# 添加 TimeSeriesTransformer 類別
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加序列維度
        x = self.transformer_encoder(x)
        return x.squeeze(1)  # 移除序列維度

# 添加 BERT 新聞處理函數
def process_news_bert(news_texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    sentiment_analyzer = pipeline('sentiment-analysis', device=0 if torch.cuda.is_available() else -1)
    model = model.to(device)
    model.eval()
    
    features = []
    sentiment_features = []
    
    for _, row in news_texts.iterrows():
        # 合併該行的所有新聞文本
        combined_text = ' '.join(row.astype(str).values)
        
        # BERT特徵提取
        inputs = tokenizer(combined_text, return_tensors='pt', 
                         max_length=512, truncation=True, padding=True)
        # 將輸入移到GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 CPU 進行 numpy 操作
            bert_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 情緒分析
        # 將文本分成較小的段落進行情緒分析
        text_chunks = [combined_text[i:i+512] for i in range(0, len(combined_text), 512)]
        chunk_sentiments = []
        
        for chunk in text_chunks:
            if len(chunk.strip()) > 0:  # 確保文本不為空
                sentiment = sentiment_analyzer(chunk)[0]
                # 將情緒標籤轉換為數值
                sentiment_score = 1.0 if sentiment['label'] == 'POSITIVE' else 0.0
                sentiment_confidence = sentiment['score']
                chunk_sentiments.append(sentiment_score * sentiment_confidence)
        
        # 計算平均情緒分數
        avg_sentiment = np.mean(chunk_sentiments) if chunk_sentiments else 0.5
        
        # 組合BERT特徵和情緒特徵
        combined_feature = np.concatenate([
            bert_feature.flatten(),
            np.array([avg_sentiment])
        ])
        
        features.append(combined_feature)
    
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
def process_djia(djia_data, window_size=5):
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
        'MACD', 'Signal_Line', 'MACD_Hist'
    ]
    
    # 使用選定的特徵
    features = df[feature_columns].values
    
    # 對每個特徵分別進行標準化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(features)
    
    return torch.FloatTensor(scaled_features), scaler

# 修改特徵融合函數
def feature_fusion(news_features, market_features):
    # 確保兩個特徵集的樣本數量相同
    min_samples = min(len(news_features), len(market_features))
    news_features = news_features[:min_samples]
    market_features = market_features[:min_samples]
    
    # 將新聞特徵和市場特徵連接起來
    fused_features = np.concatenate([news_features, market_features], axis=1)
    
    return fused_features

# 預測模型
class LateFusionPredictor(nn.Module):
    def __init__(self, news_dim, market_dim):
        super().__init__()
        self.hidden_dim = 64
        self.num_heads = 8
        self.dropout_rate = 0.3
        
        # 新聞分支編碼器
        self.news_encoder = nn.Sequential(
            nn.LayerNorm(news_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(news_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 市場分支編碼器
        self.market_encoder = nn.Sequential(
            nn.LayerNorm(market_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(market_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 時序注意力層
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # 模態注意力層
        self.modal_attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 特徵交互注意力層
        self.feature_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 預測頭
        self.news_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.market_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 最終融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Tanh()
        )
    
    def forward(self, news_x, market_x):
        try:
            # 輸入形狀: [batch_size, seq_len, feature_dim]
            batch_size, seq_len, _ = news_x.shape
            
            # 對每個時間步進行特徵編碼
            news_features = []
            market_features = []
            
            for t in range(seq_len):
                news_t = self.news_encoder(news_x[:, t, :])  # [batch_size, hidden_dim]
                market_t = self.market_encoder(market_x[:, t, :])  # [batch_size, hidden_dim]
                news_features.append(news_t)
                market_features.append(market_t)
            
            # 堆疊時序特徵
            news_features = torch.stack(news_features, dim=1)  # [batch_size, seq_len, hidden_dim]
            market_features = torch.stack(market_features, dim=1)  # [batch_size, seq_len, hidden_dim]
            
            # 應用時序注意力
            news_temporal, _ = self.temporal_attention(news_features, news_features, news_features)
            market_temporal, _ = self.temporal_attention(market_features, market_features, market_features)
            
            # 取最後一個時間步的特徵
            news_features = news_temporal[:, -1, :]  # [batch_size, hidden_dim]
            market_features = market_temporal[:, -1, :]  # [batch_size, hidden_dim]
            
            # 計算特徵重要性權重
            news_attention = self.feature_attention(news_features)  # [batch_size, 1]
            market_attention = self.feature_attention(market_features)  # [batch_size, 1]
            
            # 應用特徵重要性權重
            news_features = news_features * news_attention
            market_features = market_features * market_attention
            
            # 計算模態重要性權重
            combined_features = torch.cat([news_features, market_features], dim=-1)
            modal_weights = self.modal_attention(combined_features)  # [batch_size, 2]
            
            # 計算各分支預測
            news_pred = self.news_predictor(news_features)     # [batch_size, 1]
            market_pred = self.market_predictor(market_features)  # [batch_size, 1]
            
            # 使用模態權重融合預測結果
            predictions = torch.cat([news_pred, market_pred], dim=-1)  # [batch_size, 2]
            weighted_predictions = predictions * modal_weights  # [batch_size, 2]
            
            # 最終預測
            final_pred = self.fusion_layer(weighted_predictions)  # [batch_size, 1]
            
            return news_pred, market_pred, final_pred, modal_weights  # 返回權重以便觀察
            
        except RuntimeError as e:
            print(f"\n維度信息:")
            print(f"輸入 - 新聞: {news_x.shape}, 市場: {market_x.shape}")
            if 'news_features' in locals():
                print(f"編碼後 - 新聞特徵: {news_features.shape}")
            if 'market_features' in locals():
                print(f"編碼後 - 市場特徵: {market_features.shape}")
            if 'modal_weights' in locals():
                print(f"模態權重: {modal_weights.shape}")
            raise e

def late_fusion_loss(outputs, targets, model):
    news_pred, market_pred, fusion_pred = outputs
    
    # 確保所有預測和目標的維度一致
    targets = targets.view(-1, 1)
    
    # MSE損失
    news_loss = F.mse_loss(news_pred, targets)
    market_loss = F.mse_loss(market_pred, targets)
    fusion_loss = F.mse_loss(fusion_pred, targets)
    
    # 波動性損失 - 確保預測值的波動與實際值相似
    if len(fusion_pred) > 1:
        pred_volatility = torch.std(fusion_pred)
        target_volatility = torch.std(targets)
        volatility_loss = F.mse_loss(pred_volatility, target_volatility)
        
        # 計算預測值和實際值的一階差分
        pred_diff = fusion_pred[1:] - fusion_pred[:-1]
        target_diff = targets[1:] - targets[:-1]
        
        # 差分MSE損失
        diff_loss = F.mse_loss(pred_diff, target_diff)
        
        # 方向預測損失（使用平滑版本）
        direction_match = torch.sigmoid(pred_diff * target_diff * 10)
        direction_loss = -torch.mean(direction_match)
    else:
        volatility_loss = torch.tensor(0.0, device=targets.device)
        diff_loss = torch.tensor(0.0, device=targets.device)
        direction_loss = torch.tensor(0.0, device=targets.device)
    
    # 相關性損失 - 確保預測值與實際值的相關性
    if len(fusion_pred) > 1:
        pred_normalized = (fusion_pred - fusion_pred.mean()) / fusion_pred.std()
        target_normalized = (targets - targets.mean()) / targets.std()
        correlation_loss = -torch.mean(pred_normalized * target_normalized)
    else:
        correlation_loss = torch.tensor(0.0, device=targets.device)
    
    # L2正則化
    l2_loss = sum(p.pow(2.0).sum() for p in model.parameters()) * 0.0001
    
    # 總損失
    total_loss = (
        0.5 * fusion_loss +      # 主要預測損失
        0.1 * news_loss +        # 新聞分支損失
        0.1 * market_loss +      # 市場分支損失
        0.05 * volatility_loss + # 波動性損失
        0.05 * diff_loss +       # 差分損失
        0.05 * direction_loss +   # 方向損失
        0.05 * correlation_loss + # 相關性損失
        0.001 * l2_loss          # L2正則化
    )
    
    return total_loss

# 修改主要處理流程
def main():
    # 訓練參數設置
    num_epochs = 200
    batch_size = 16   # 減小批次大小
    seq_len = 10
    
    # 1. 處理新聞文本
    news_features = process_news_bert(news_data.iloc[:, 2:])
    
    # 2. 處理 DJIA 數據
    market_features, price_scaler = process_djia(djia_data, window_size=seq_len)
    
    # 確保數據長度匹配
    min_len = min(len(news_features), len(market_features))
    news_features = news_features[:min_len]
    market_features = market_features[:min_len]
    
    # 保存日期資訊並按時間順序排序
    dates = pd.to_datetime(djia_data['Date'])[:min_len]
    sort_idx = dates.argsort()
    dates = dates.iloc[sort_idx]
    news_features = news_features[sort_idx]
    market_features = market_features[sort_idx]
    
    # 準備標籤
    labels = djia_data['Close'].values[sort_idx][:min_len]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))
    
    # 創建時序序列數據
    def create_sequences(features, labels, seq_length):
        sequences = []
        targets = []
        for i in range(len(features) - seq_length):
            seq = features[i:(i + seq_length)]
            target = labels[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    # 創建時序序列
    X_news_sequences, _ = create_sequences(news_features, labels_scaled, seq_len)
    X_market_sequences, y_sequences = create_sequences(market_features, labels_scaled, seq_len)
    
    # 分割訓練集和測試集（保持時間順序）
    train_size = int(0.8 * len(X_news_sequences))
    
    X_news_train = X_news_sequences[:train_size]
    X_news_test = X_news_sequences[train_size:]
    X_market_train = X_market_sequences[:train_size]
    X_market_test = X_market_sequences[train_size:]
    y_train = y_sequences[:train_size]
    y_test = y_sequences[train_size:]
    
    # 轉換為PyTorch張量
    X_news_train = torch.FloatTensor(X_news_train).to(device)
    X_news_test = torch.FloatTensor(X_news_test).to(device)
    X_market_train = torch.FloatTensor(X_market_train).to(device)
    X_market_test = torch.FloatTensor(X_market_test).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # 打印序列形狀
    print(f"\n數據形狀:")
    print(f"新聞序列訓練集: {X_news_train.shape}")
    print(f"市場序列訓練集: {X_market_train.shape}")
    print(f"標籤訓練集: {y_train.shape}")
    
    # 初始化模型
    news_dim = news_features.shape[1]
    market_dim = market_features.shape[1]
    predictor = LateFusionPredictor(
        news_dim=news_dim,
        market_dim=market_dim
    ).to(device)
    
    # 優化器設置
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=0.0001,           # 降低基礎學習率
        weight_decay=0.01    # 增加權重衰減
    )
    
    # 計算總步數
    steps_per_epoch = (len(X_news_train) + batch_size - 1) // batch_size
    
    # 學習率調度
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,        # 降低最大學習率
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,       # 減少預熱期
        anneal_strategy='cos',
        div_factor=10.0,     # 減小學習率範圍
        final_div_factor=100.0
    )
    
    # 訓練循環
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        predictor.train()
        epoch_loss = 0
        num_batches = 0
        
        # 訓練階段
        for i in range(0, len(X_news_train), batch_size):
            end_idx = min(i + batch_size, len(X_news_train))
            batch_news = X_news_train[i:end_idx]
            batch_market = X_market_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            
            optimizer.zero_grad()
            news_pred, market_pred, fusion_pred, modal_weights = predictor(batch_news, batch_market)
            loss = late_fusion_loss((news_pred, market_pred, fusion_pred), batch_y, predictor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        
        # 驗證階段
        predictor.eval()
        with torch.no_grad():
            news_pred, market_pred, fusion_pred, modal_weights = predictor(X_news_test, X_market_test)
            test_loss = late_fusion_loss((news_pred, market_pred, fusion_pred), y_test, predictor)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] - '
                  f'Train Loss: {avg_epoch_loss:.6f}, '
                  f'Test Loss: {test_loss:.6f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # 早停檢查
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                patience_counter = 0
                best_model_state = predictor.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                predictor.load_state_dict(best_model_state)
                break
    
    # 訓練完成後進行預測和視覺化
    predictor.eval()
    with torch.no_grad():
        news_pred, market_pred, fusion_pred, modal_weights = predictor(X_news_test, X_market_test)
        
        # 轉換回原始價格範圍
        fusion_pred_np = fusion_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # 使用價格縮放器進行反轉換
        pred_prices = scaler.inverse_transform(fusion_pred_np)
        true_prices = scaler.inverse_transform(y_test_np)
        
        # 計算評估指標
        mse = mean_squared_error(true_prices, pred_prices)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_prices - pred_prices))
        mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
        
        # 創建時間序列
        test_dates = dates[-len(y_test):]
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建圖表
        plt.figure(figsize=(15, 10))
        
        # 繪製預測結果
        plt.plot(test_dates, true_prices, label='實際價格', color='blue', linewidth=2)
        plt.plot(test_dates, pred_prices, label='預測價格', color='red', linestyle='--', linewidth=2)
        
        # 設置標題和標籤
        plt.title('道瓊斯指數預測結果', fontsize=16, pad=20)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('價格', fontsize=12)
        
        # 添加圖例
        plt.legend(fontsize=12)
        
        # 旋轉x軸標籤
        plt.xticks(rotation=45)
        
        # 添加網格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加評估指標文字
        metrics_text = f'評估指標:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%'
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=12,
                verticalalignment='top')
        
        # 調整佈局
        plt.tight_layout()
        
        # 保存圖片
        save_path = "C:\\Users\\User\\Desktop\\金融期末\\晚期融合\\prediction_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n預測結果圖表已保存至: {save_path}")
        
        # 關閉圖表
        plt.close()
        
        # 打印評估指標
        print(f"\n模型評估指標:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
    
    return predictor, rmse

if __name__ == "__main__":
    model, rmse = main()
    print(f"\n模型訓練完成，最終 RMSE: {rmse:.2f}")