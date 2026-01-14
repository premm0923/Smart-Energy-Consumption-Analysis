from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

print("Loading Website Model...")
model = tf.keras.models.load_model('website_model.h5', compile=False)
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- DEBUGGING: PRINT EXACTLY WHAT WE RECEIVED ---
        print("------------------------------------------------")
        print(f"INCOMING DATA: {request.form}") 
        print("------------------------------------------------")

        # --- THE UNIVERSAL FIX ---
        # We try to find the input under 'prev_usage'. 
        # If that fails, we look for 'usage'.
        # This handles both old and new HTML versions.
        
        user_input_str = request.form.get('prev_usage')
        if user_input_str is None:
             user_input_str = request.form.get('usage')
        
        # If STILL None, then the form is truly empty
        if user_input_str is None:
            return "Error: The form sent no data! Refresh your page.", 400

        # Convert to float once we found it
        prev_usage = float(user_input_str)
        day_of_week = int(request.form['day_of_week']) 
        
        # Auto-fill the rest
        current_hour = 12.0
        day_of_month = 15.0
        month = 6.0
        rolling_mean = prev_usage 

        features = [current_hour, float(day_of_week), day_of_month, month, prev_usage, prev_usage, rolling_mean]
        
        final_features = np.array([features])
        scaled_features = scaler.transform(final_features)
        lstm_input = scaled_features.reshape((1, 1, 7))
        
        pred_next_hour = model.predict(lstm_input)[0][0]
        pred_next_hour = max(0, pred_next_hour) 

        pred_3_hours = pred_next_hour * 3
        pred_next_week = rolling_mean * 24 * 7
        pred_next_month = rolling_mean * 24 * 30

        return render_template('index.html', 
                               p_1h=f'{pred_next_hour:.4f}',
                               p_3h=f'{pred_3_hours:.2f}',
                               p_week=f'{pred_next_week:.2f}',
                               p_month=f'{pred_next_month:.2f}',
                               show_result=True)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return render_template('index.html', error_text=f'Server Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)