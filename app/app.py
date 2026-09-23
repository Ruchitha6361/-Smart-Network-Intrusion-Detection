from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(
    __name__,
    template_folder=r"D:\Cybersecurity Projects\Smart-Network-Intrusion-Detection\templates"
)

# Load trained ML model
model = joblib.load(
    r"D:\Cybersecurity Projects\Smart-Network-Intrusion-Detection\models\intrusion_detection_model.pkl"
)


# MySQL connection
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD"),
        database="intrusion_detection",
        cursorclass=pymysql.cursors.DictCursor
    )


@app.route("/")
def home():
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC"
            )
            predictions = cursor.fetchall()

            cursor.execute(
                "SELECT COUNT(*) AS total FROM predictions"
            )
            total = cursor.fetchone()["total"]

            cursor.execute(
                "SELECT COUNT(*) AS attacks FROM predictions WHERE prediction = 1"
            )
            attacks = cursor.fetchone()["attacks"]

            cursor.execute(
                "SELECT COUNT(*) AS normal FROM predictions WHERE prediction = 0"
            )
            normal = cursor.fetchone()["normal"]

    finally:
        connection.close()

    return render_template(
        "index.html",
        predictions=predictions,
        total=total,
        attacks=attacks,
        normal=normal
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]

        result = (
            "Attack Detected"
            if prediction == 1
            else "Normal Traffic"
        )

        connection = get_db_connection()

        try:
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO predictions (prediction, result)
                    VALUES (%s, %s)
                """
                cursor.execute(
                    sql,
                    (int(prediction), result)
                )

            connection.commit()

        finally:
            connection.close()

        return jsonify({
            "prediction": int(prediction),
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


@app.route("/history")
def history():
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC"
            )
            predictions = cursor.fetchall()

        return jsonify(predictions)

    finally:
        connection.close()


if __name__ == "__main__":
    app.run(debug=False)