from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_data = {k: float(v) if v.replace('.', '', 1).isdigit() else v for k, v in data.items()}

        custom_data = CustomData(
            funding_total_usd=input_data["funding_total_usd"],
            funding_rounds=input_data["funding_rounds"],
            avg_participants=input_data["avg_participants"],
            relationships=input_data["relationships"],
            milestones=input_data["milestones"],
            has_angel=input_data["has_angel"],
            has_VC=input_data["has_VC"],
            has_roundA=input_data["has_roundA"],
            has_roundB=input_data["has_roundB"],
            has_roundC=input_data["has_roundC"],
            has_roundD=input_data["has_roundD"],
            is_CA=input_data["is_CA"],
            is_NY=input_data["is_NY"],
            is_MA=input_data["is_MA"],
            is_TX=input_data["is_TX"],
            is_otherstate=input_data["is_otherstate"],
            is_software=input_data["is_software"],
            is_web=input_data["is_web"],
            is_mobile=input_data["is_mobile"],
            is_enterprise=input_data["is_enterprise"],
            is_advertising=input_data["is_advertising"],
            is_gamesvideo=input_data["is_gamesvideo"],
            is_ecommerce=input_data["is_ecommerce"],
            is_consulting=input_data["is_consulting"],
            is_othercategory=input_data["is_othercategory"],
            is_top500=input_data["is_top500"],
            age_first_funding_year=input_data["age_first_funding_year"],
            age_last_funding_year=input_data["age_last_funding_year"],
            age_first_milestone_year=input_data["age_first_milestone_year"],
            age_last_milestone_year=input_data["age_last_milestone_year"],
            state_code=input_data["state_code"],
            category_code=input_data["category_code"]
        )

        df = custom_data.get_data_as_data_frame()
        prediction = PredictPipeline().predict(df)[0]

        label_map = {0: "Bankrupt", 1: "Operating"}
        predicted_label = label_map.get(prediction, "Unknown")

        return render_template("index.html", prediction=predicted_label)

    except Exception as e:
        return render_template("index.html", prediction=f"⚠️ Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
