document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    // Show loading spinner
    document.getElementById('loading').style.display = 'block';

    const formData = new FormData(this);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    // Hide loading spinner
    document.getElementById('loading').style.display = 'none';

    // Display result
    document.getElementById('result').innerHTML = `
        <div class="result-card">
            <h2>Prediction Result</h2>
            <p><strong>Final Weight (lbs):</strong> ${result['Final Weight (lbs)']} lbs</p>
            <p><strong>Mean Squared Error (MSE):</strong> ${result.MSE}</p>
            <p><strong>Mean Absolute Error (MAE):</strong> ${result.MAE}</p>
            <p><strong>R² Score:</strong> ${result['R² Score']}</p>
            <p><strong>Confidence:</strong> ${result.Confidence}</p>
        </div>
    `;
});

function convertLbsToKg() {
    const lbs = parseFloat(document.getElementById('lbs_input').value);
    if (!isNaN(lbs)) {
        const kg = (lbs / 2.20462).toFixed(2);
        document.getElementById('kg_output').value = kg;
    } else {
        document.getElementById('kg_output').value = '';
    }
}

function convertKgToLbs() {
    const kg = parseFloat(document.getElementById('kg_input').value);
    if (!isNaN(kg)) {
        const lbs = (kg * 2.20462).toFixed(2);
        document.getElementById('lbs_output').value = lbs;
    } else {
        document.getElementById('lbs_output').value = '';
    }
}
