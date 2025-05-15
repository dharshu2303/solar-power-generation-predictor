document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('results-section');
    const errorMessage = document.getElementById('error-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    const submitText = document.getElementById('submit-text');
    const newPredictionBtn = document.getElementById('new-prediction-btn');
    let predictionChart;

    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        loadingSpinner.style.display = 'inline-block';
        submitText.textContent = 'Loading...';
        errorMessage.classList.add('d-none');
        const city = document.getElementById('cityInput').value.trim();
    
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ city: city })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to get prediction');
                });
            }
            return response.json();
        })
        .then(data => {
            loadingSpinner.style.display = 'none';
            submitText.textContent = 'Predict Solar Output';

            displayResults(data);
        })
        .catch(error => {
            loadingSpinner.style.display = 'none';
            submitText.textContent = 'Predict Solar Output';
            errorMessage.textContent = error.message;
            errorMessage.classList.remove('d-none');
            resultsSection.style.display = 'none';
        });
    });

    newPredictionBtn.addEventListener('click', function() {
        resultsSection.style.display = 'none';
        predictionForm.reset();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    function displayResults(data) {
        document.getElementById('location-header').textContent = `${data.weather.city} (${data.weather.latitude.toFixed(2)}°N, ${data.weather.longitude.toFixed(2)}°E)`;
        document.getElementById('current-time').textContent = data.weather.timestamp;
        document.getElementById('weather-description').textContent = data.weather.weather_description;
        document.getElementById('temperature').textContent = `${data.weather.temperature_2_m_above_gnd.toFixed(1)}°C`;
        document.getElementById('cloud-cover').textContent = `${data.weather.total_cloud_cover_sfc}% Cloud Cover`;
        document.getElementById('wind-speed').textContent = `${data.weather.wind_speed_10_m_above_gnd} m/s Wind Speed`;
        document.getElementById('power-output').textContent = `${data.prediction} kW`;

        const tipsContainer = document.getElementById('tips-container');
        tipsContainer.innerHTML = '';
        
        data.tips.forEach(tip => {
            const tipCol = document.createElement('div');
            tipCol.className = 'col-md-6';
            const tipCard = document.createElement('div');
            tipCard.className = 'card tip-card p-3 mb-3';
            tipCard.textContent = tip;
            tipCol.appendChild(tipCard);
            tipsContainer.appendChild(tipCol);
        });
        initChart(data);
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function initChart(data) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        if (predictionChart) {
            predictionChart.destroy();
        }
        const currentHour = new Date(data.weather.timestamp).getHours();
        const labels = [];
        const predictions = [];
        
        for (let i = 0; i < 24; i++) {
            labels.push(`${i}:00`);
            let factor = 0;
            if (i >= 6 && i <= 18) {
                factor = 1 - Math.pow((i - 12) / 6, 2);
                factor = Math.max(0, factor);
            }
            
            const cloudImpact = 1 - (data.weather.total_cloud_cover_sfc / 100) * 0.7;
            const hourPrediction = data.prediction * factor * cloudImpact;
            predictions.push(hourPrediction.toFixed(2));
        }
        
        predictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Estimated Power Output (kW)',
                    data: predictions,
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                    pointBorderColor: '#fff',
                    pointRadius: function(context) {
                        const index = context.dataIndex;
                        return index === currentHour ? 6 : 3;
                    },
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Hour: ${tooltipItems[0].label}`;
                            },
                            label: function(context) {
                                return `Power: ${context.raw} kW`;
                            }
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Hour of Day'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Power (kW)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}); 
