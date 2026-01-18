from flask import Flask, render_template, request, redirect, url_for, jsonify
import pymrio_225 as pyn
import string
import time

app = Flask(__name__)
shock = None
country = None
industry = None

@app.route('/', methods = ['GET', 'POST'])
def home():
    start_time = time.time();

    if request.method == 'POST':
        try:
            new_shock = int(request.form['shockFactor'])
            pyn.update_shock_factor(new_shock);
        except ValueError:
            pass

    print("Calculating results...")
    results = pyn.calculate_results()
    elapsed = time.time() - start_time
    print(f"Result calc time: {elapsed:.2f} seconds")

        
    display = {};

    for key,value in results.items():
        
        if key == ('USA', 'Agriculture'):
            display[key] = value;
        

    return render_template('interface_2.4.html', shock=pyn.shockFactor, country=pyn.shockCountry, industry = pyn.shockIndustry, results = display);

@app.route('/updateShock', methods = ['POST'])
def updateShock():
    data = request.get_json()
    shock = float(data["shock"])
    country = data["country"]
    industry = data["industry"]

    try:
        shock_val = int(shock)
        pyn.update_shock_factor(shock_val, country, industry)

        # Get original and shocked results
        original = pyn.originalOutput
        shocked = pyn.calculate_results()

        # Top 5 impacted sectors by absolute change
        top_items = sorted(shocked.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        labels = [f"{k[0]}: {k[1]}" for k, _ in top_items]
        original_values = [original[k] for k, _ in top_items]
        shocked_values = [original[k] + shocked[k] for k, _ in top_items]

        return jsonify({
            "status": "success",
            "labels": labels,
            "original": original_values,
            "shocked": shocked_values
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400





if __name__ == '__main__':
    print("starting server")
    app.run(debug=True)
