### Gerekli Kütüphanelerin Import edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

# Read the data
df = pd.read_excel('Veri-Seti.xlsx')
###,index_col=[0],parse_dates=[0] 
df = df.dropna()
df.set_index('Date', inplace=True)

# Haziran 2022'den başlayarak 12 ay boyunca tarihler oluşturuyoruz
date_range = pd.date_range(start='2022-06-01', end='2023-06-01', freq='MS')
# oluşturduğumuz tarihleri kullanarak bir dataframe oluşturuyoruz
forecast_df = pd.DataFrame({'Date': date_range})

# Create the seasonal variables
df['Month'] = df.index.month
dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True)
df = pd.concat([df, dummies], axis=1)
df['Season_JJA'] = ((df['Month'] == 6) | (df['Month'] == 7) | (df['Month'] == 8)).astype(int)

# Split the data into training and test sets
X = df.drop(['Otomotiv Satis', 'Month'], axis=1)
y = df['Otomotiv Satis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

## Prediction with real variables
car_price_predictions = model.predict(X_test)

### Prediction with error analysis
y_pred = car_price_predictions
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
lm_score=model.score(X_test,y_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("determination of prediction Score:",lm_score)

#Actual value and the predicted value difference
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
print(mlr_diff.head())

# Grafiği çizme
print("Gerçek Veriler ile Tahmin Verileri Karşılaştırma ")
mlr_diff_1=mlr_diff
mlr_diff_1.sort_index(inplace=True)

# Grafiği çizme
plt.plot(mlr_diff_1.index, mlr_diff_1['Actual value'], color="yellowgreen",label='Gerçek Veriler')
plt.plot(mlr_diff_1.index, mlr_diff_1['Predicted value'], label='Tahmin Verileri')
plt.xlabel('Tarhileri')
plt.ylabel('Otomativ Satışları')
plt.title('Zaman İçinde Gerçekleşen ve Tahmin Edilen Değerleri')
plt.savefig('Zaman İçinde Gerçekleşen ve Tahmin Edilen Değerleri.png')



# Serialize the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Haziran 2022'den başlayarak 12 ay boyunca tarihler oluşturuyoruz
date_range = pd.date_range(start='2022-06-01', end='2023-06-01', freq='MS')
# oluşturduğumuz tarihleri kullanarak bir dataframe oluşturuyoruz
forecast_df_2 = pd.DataFrame({'Date': date_range})

# Verilerin istatistiksel davranışlarını kullanarak rastgele değerler oluşturuyoruz
otv_values = np.random.randint(low=37, high=65, size=len(forecast_df_2))
kredi_values = np.random.normal(loc=1787554.29, scale=1176365.97, size=len(forecast_df_2))
faiz_values = np.random.normal(loc=16.03, scale=5.32, size=len(forecast_df_2))
eur_values = np.random.normal(loc=4.81, scale=3.32, size=len(forecast_df_2))

# Oluşturduğumuz rastgele değerleri dataframe'e ekliyoruz
forecast_df_2['OTV Orani'] = otv_values
forecast_df_2['Kredi Stok'] = kredi_values
forecast_df_2['Faiz'] = faiz_values
forecast_df_2['EUR/TL'] = eur_values

# Oluşturğum rastgele değerlerin mevsimsel değerlerin eklenmesi
forecast_df_2['Month'] = forecast_df_2['Date'].dt.month
dummies = pd.get_dummies(forecast_df_2['Month'], prefix='Month', drop_first=True)
forecast_df_2 = pd.concat([forecast_df_2, dummies], axis=1)
forecast_df_2['Season_JJA'] = ((forecast_df_2['Month'] == 6) | (forecast_df_2['Month'] == 7) | (forecast_df_2['Month'] == 8)).astype(int)


# Özellikleri ölçeklendiriyoruz
#forecast_df_2 = forecast_df_2.drop(['Predicted Otomotiv Satis', 'Month'], axis=1)
forecast_df_scaled_2 =ss.transform(forecast_df_2[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'Month_2', 'Month_3',
       'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9',
       'Month_10', 'Month_11', 'Month_12', 'Season_JJA']])



print("Forecast With Linear Regression Model " )
# Tahmin yaparak sonuçları 'Predicted Otomotiv Satis' sütununda saklıyoruz
forecast_df_2['Predicted Otomotiv Satis'] = model.predict(forecast_df_scaled_2)

# Tahmin sonuçlarını yazdırıyoruz
print("Haz’22 – Haz’23 dönemleri için tahminler:")
forecast_df_1= forecast_df_2[['Date','Predicted Otomotiv Satis']]
print(forecast_df_1)


#### Tahmin sonuçlarını görselliyoruz
# Grafiği çizme
# Grafiği çizme
plt.figure(figsize=(12,8))
sns.set_style("whitegrid")
plt.plot(forecast_df_1['Date'],forecast_df_1[ 'Predicted Otomotiv Satis'],color='darkviolet',linewidth=3.5)
plt.xticks(rotation=45, fontsize=10)
plt.title('Otomotiv Satışı Tahminleri', fontsize=14,fontweight='bold')
plt.xlabel('Tarih', fontsize=12,fontweight='bold')
plt.ylabel('Otomotiv Satış', fontsize=12,fontweight='bold')
plt.savefig("Otamativ Satış Tahminleri")
#plt.show()

#### Tahmin sonuçlarını görselliyoruz
plt.figure(figsize=(12,8))
plt.plot(df.index, df['Otomotiv Satis'], label='Gerçek Değerler')
plt.plot(mlr_diff_1.index, mlr_diff_1['Predicted value'],color="yellowgreen", label='Model Tahminleri')
plt.plot(forecast_df_2['Date'],forecast_df_2[ 'Predicted Otomotiv Satis'],color='darkviolet', label="Forecast Tahminleri ")
plt.title('Otomotiv Satış Verileri ve Tahminleri')
plt.xlabel('Tarihler')
plt.ylabel('Otomotiv Satış Sayıları')
plt.savefig('Otomotiv Satış Verileri ve Forecast Tahminleri.png')

####Flask bağlantısı ile postman bağlantısı ile sorgu gönderme

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/prediction', methods=['GET'])
def prediction():
    date = request.args.get('date')
    result = forecast_df_1[forecast_df_1['Date'] == date]['Predicted Otomotiv Satis'].values[0]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

