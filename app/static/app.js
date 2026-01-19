/**
 * GoldAI Pro - Frontend Application
 * ==================================
 * Professional gold price forecasting dashboard
 */

// Configuration
const API_BASE = '';
const UPDATE_INTERVAL = 60000; // 1 minute

// State
let priceChart = null;
let currentDays = 90;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    console.log('🚀 Initializing GoldAI Pro...');
    
    // Load all data
    await Promise.all([
        fetchCurrentPrice(),
        fetchPrediction(),
        fetchModelMetrics(),
        fetchFeatureImportance(),
        fetchHistoricalData(currentDays),
        fetchRecommendation()
    ]);
    
    // Setup event listeners
    setupEventListeners();
    
    // Setup auto-refresh
    setInterval(fetchCurrentPrice, UPDATE_INTERVAL);
    setInterval(fetchPrediction, UPDATE_INTERVAL * 5);
    setInterval(fetchRecommendation, UPDATE_INTERVAL * 5);
    
    console.log('✅ GoldAI Pro initialized');
}

function setupEventListeners() {
    // Time period buttons
    document.querySelectorAll('.btn-time').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-time').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentDays = parseInt(e.target.dataset.days);
            fetchHistoricalData(currentDays);
        });
    });
}

// API Functions
async function fetchCurrentPrice() {
    try {
        const response = await fetch(`${API_BASE}/api/current-price`);
        const data = await response.json();
        
        if (data.success) {
            updatePriceDisplay(data);
        }
    } catch (error) {
        console.error('Error fetching current price:', error);
    }
}

async function fetchPrediction() {
    try {
        const response = await fetch(`${API_BASE}/api/predict`);
        const data = await response.json();
        
        if (data.success) {
            updatePredictionDisplay(data);
        }
    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

async function fetchModelMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/model-metrics`);
        const data = await response.json();
        
        if (data.success) {
            updateMetricsDisplay(data.metrics);
        }
    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
}

async function fetchFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/api/feature-importance`);
        const data = await response.json();
        
        if (data.success) {
            updateFeatureImportance(data.importance);
        }
    } catch (error) {
        console.error('Error fetching feature importance:', error);
    }
}

async function fetchRecommendation() {
    try {
        const response = await fetch(`${API_BASE}/api/recommendation`);
        const data = await response.json();
        
        if (data.success) {
            updateRecommendationDisplay(data);
        }
    } catch (error) {
        console.error('Error fetching recommendation:', error);
    }
}

async function fetchHistoricalData(days) {
    try {
        const response = await fetch(`${API_BASE}/api/historical?days=${days}`);
        const data = await response.json();
        
        if (data.success) {
            updateChart(data.data);
        }
    } catch (error) {
        console.error('Error fetching historical data:', error);
    }
}

async function refreshAllData() {
    const btn = document.querySelector('.btn-refresh i');
    btn.style.animation = 'spin 1s linear infinite';
    
    try {
        await fetch(`${API_BASE}/api/refresh-data`);
        await Promise.all([
            fetchCurrentPrice(),
            fetchPrediction(),
            fetchHistoricalData(currentDays)
        ]);
    } finally {
        setTimeout(() => {
            btn.style.animation = '';
        }, 1000);
    }
}

// UI Update Functions
function updatePriceDisplay(data) {
    // Current price
    document.getElementById('currentPrice').textContent = formatPrice(data.price);
    
    // Price change
    const changeEl = document.getElementById('priceChange');
    const changeValue = data.change >= 0 ? `+${formatPrice(data.change)}` : formatPrice(data.change);
    const changePercent = data.change_pct >= 0 ? `+${data.change_pct.toFixed(2)}%` : `${data.change_pct.toFixed(2)}%`;
    
    changeEl.innerHTML = `
        <i class="fas fa-${data.change >= 0 ? 'caret-up' : 'caret-down'}"></i>
        <span class="change-value">${changeValue}</span>
        <span class="change-percent">(${changePercent})</span>
    `;
    changeEl.className = `price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
    
    // Price details
    document.getElementById('priceOpen').textContent = formatPrice(data.open);
    document.getElementById('priceHigh').textContent = formatPrice(data.high);
    document.getElementById('priceLow').textContent = formatPrice(data.low);
}

function updatePredictionDisplay(data) {
    // Predicted price (next day)
    document.getElementById('predictedPrice').textContent = formatPrice(data.prediction);
    
    // Direction
    const dirEl = document.getElementById('predictionDirection');
    dirEl.className = `prediction-direction ${data.direction}`;
    dirEl.innerHTML = `<i class="fas fa-arrow-${data.direction === 'up' ? 'trend-up' : 'trend-down'}"></i>`;
    
    // Price Change Display
    const changeEl = document.getElementById('predictionChange');
    if (changeEl) {
        const isPositive = data.change >= 0;
        const changeSign = isPositive ? '+' : '';
        const arrowIcon = isPositive ? 'fa-caret-up' : 'fa-caret-down';
        
        changeEl.className = `prediction-change-display ${isPositive ? 'positive' : 'negative'}`;
        changeEl.innerHTML = `
            <span class="change-arrow"><i class="fas ${arrowIcon}"></i></span>
            <span class="change-amount">${changeSign}$${Math.abs(data.change).toFixed(2)}</span>
            <span class="change-percent">(${changeSign}${data.change_pct.toFixed(2)}%)</span>
        `;
    }
    
    // Confidence - Update ring animation
    const confidence = data.confidence > 1 ? data.confidence : (data.confidence * 100);
    document.getElementById('confidenceValue').textContent = `${confidence.toFixed(1)}%`;
    
    // Animate the confidence ring (circumference = 2 * π * 45 ≈ 283)
    const circumference = 283;
    const offset = circumference - (confidence / 100) * circumference;
    const ringEl = document.getElementById('confidenceRing');
    if (ringEl) {
        ringEl.style.strokeDashoffset = offset;
        // Set stroke color based on confidence level
        if (confidence >= 95) {
            ringEl.style.stroke = '#00d4aa'; // Green for high confidence
        } else if (confidence >= 80) {
            ringEl.style.stroke = '#ffd700'; // Gold for medium-high
        } else {
            ringEl.style.stroke = '#ff9500'; // Orange for lower
        }
    }
    
    // Update multi-timeframe predictions
    if (data.predictions) {
        updateTimeframePrediction('pred3M', data.predictions['3M']);
        updateTimeframePrediction('pred6M', data.predictions['6M']);
        updateTimeframePrediction('pred1Y', data.predictions['1Y']);
    }
}

function updateTimeframePrediction(elementId, predData) {
    const card = document.getElementById(elementId);
    if (!card || !predData) return;
    
    // Update card class based on direction
    card.className = `timeframe-card ${predData.direction === 'up' ? 'bullish' : 'bearish'}`;
    
    // Update price - format with commas
    const priceEl = card.querySelector('.timeframe-price');
    if (priceEl) {
        priceEl.textContent = `$${formatPrice(predData.price)}`;
    }
    
    // Update change with arrow icon
    const changeEl = card.querySelector('.timeframe-change');
    if (changeEl) {
        const changeSign = predData.change >= 0 ? '+' : '';
        const arrow = predData.change >= 0 ? '↑' : '↓';
        changeEl.innerHTML = `${arrow} ${changeSign}${predData.change_pct.toFixed(1)}%`;
        changeEl.className = `timeframe-change ${predData.change >= 0 ? 'positive' : 'negative'}`;
    }
    
    // Update confidence
    const confEl = card.querySelector('.timeframe-confidence');
    if (confEl) {
        confEl.textContent = `${predData.confidence.toFixed(1)}% confidence`;
    }
}

function updateMetricsDisplay(metrics) {
    // Display high-accuracy metrics
    const accuracy = metrics.accuracy || metrics.directional_accuracy;
    document.getElementById('metricAccuracy').textContent = `${(accuracy * 100).toFixed(2)}%`;
    document.getElementById('metricDirAcc').textContent = `${(metrics.directional_accuracy * 100).toFixed(2)}%`;
    document.getElementById('metricR2').textContent = metrics.r_squared ? `${(metrics.r_squared * 100).toFixed(2)}%` : '--';
    document.getElementById('metricSharpe').textContent = metrics.sharpe_ratio.toFixed(2);
    
    // Add visual indicator for high accuracy
    if (accuracy >= 0.99) {
        document.getElementById('metricAccuracy').classList.add('accuracy-excellent');
        document.getElementById('metricDirAcc').classList.add('accuracy-excellent');
    }
}

function updateFeatureImportance(importance) {
    const container = document.getElementById('featureImportance');
    const maxImportance = Math.max(...Object.values(importance));
    
    let html = '';
    Object.entries(importance)
        .sort((a, b) => b[1] - a[1])
        .forEach(([feature, value]) => {
            const width = (value / maxImportance * 100).toFixed(1);
            html += `
                <div class="feature-bar">
                    <span class="feature-name">${feature}</span>
                    <div class="feature-progress">
                        <div class="feature-fill" style="width: ${width}%">
                            <span class="feature-value">${(value * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `;
        });
    
    container.innerHTML = html;
}

function updateRecommendationDisplay(data) {
    const recommendation = data.recommendation.toLowerCase();
    
    // Update signal icon and text
    const signalContainer = document.getElementById('recommendationSignal');
    const signalIcon = signalContainer.querySelector('.signal-icon');
    const signalText = signalContainer.querySelector('.signal-text');
    
    // Set icon based on recommendation
    let iconClass, iconName;
    if (recommendation === 'buy') {
        iconClass = 'buy';
        iconName = 'fa-arrow-trend-up';
    } else if (recommendation === 'sell') {
        iconClass = 'sell';
        iconName = 'fa-arrow-trend-down';
    } else {
        iconClass = 'hold';
        iconName = 'fa-hand';
    }
    
    signalIcon.className = `signal-icon ${iconClass}`;
    signalIcon.innerHTML = `<i class="fas ${iconName}"></i>`;
    signalText.className = `signal-text ${iconClass}`;
    signalText.textContent = data.recommendation;
    
    // Update confidence meter
    const confidenceBar = document.getElementById('recommendationConfidence');
    const confidenceValue = document.getElementById('recommendationConfidenceValue');
    const confidence = (data.confidence * 100).toFixed(0);
    
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.className = `meter-fill ${iconClass}`;
    confidenceValue.textContent = `${confidence}%`;
    
    // Update action text
    document.getElementById('actionText').textContent = data.action_text;
    
    // Update summary
    document.getElementById('recommendationSummary').innerHTML = `<p>${data.summary}</p>`;
    
    // Update factors
    const factorsGrid = document.getElementById('factorsGrid');
    let factorsHtml = '';
    
    data.factors.forEach(factor => {
        factorsHtml += `
            <div class="factor-card ${factor.impact}">
                <div class="factor-header">
                    <span class="factor-name">${factor.factor}</span>
                    <span class="factor-badge ${factor.impact}">${factor.impact}</span>
                </div>
                <p class="factor-detail">${factor.detail}</p>
            </div>
        `;
    });
    
    factorsGrid.innerHTML = factorsHtml;
    
    // Update technical snapshot
    const tech = data.technical_data;
    
    document.getElementById('techRSI').textContent = tech.rsi.toFixed(1);
    document.getElementById('techRSI').className = `snapshot-value ${tech.rsi > 70 ? 'negative' : tech.rsi < 30 ? 'positive' : ''}`;
    
    const return5d = tech.returns_5d;
    document.getElementById('techReturn5d').textContent = `${return5d >= 0 ? '+' : ''}${return5d.toFixed(2)}%`;
    document.getElementById('techReturn5d').className = `snapshot-value ${return5d >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('techVolatility').textContent = `${tech.volatility.toFixed(2)}%`;
    
    document.getElementById('techMA5').textContent = `$${tech.ma_5.toFixed(2)}`;
    document.getElementById('techMA20').textContent = `$${tech.ma_20.toFixed(2)}`;
    
    const return20d = tech.returns_20d;
    document.getElementById('techReturn20d').textContent = `${return20d >= 0 ? '+' : ''}${return20d.toFixed(2)}%`;
    document.getElementById('techReturn20d').className = `snapshot-value ${return20d >= 0 ? 'positive' : 'negative'}`;
    
    // Update disclaimer
    if (data.disclaimer) {
        document.getElementById('disclaimerText').textContent = data.disclaimer;
    }
}

function updateChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }
    
    // Calculate prediction line (extend last few points)
    const predictionDates = [...data.dates.slice(-10)];
    const predictionPrices = [...data.prices.slice(-10)];
    
    // Add future prediction point
    const lastDate = new Date(data.dates[data.dates.length - 1]);
    lastDate.setDate(lastDate.getDate() + 1);
    predictionDates.push(lastDate.toISOString().split('T')[0]);
    
    // Simple trend projection for demo
    const lastPrice = data.prices[data.prices.length - 1];
    const trend = (data.prices[data.prices.length - 1] - data.prices[data.prices.length - 5]) / 5;
    predictionPrices.push(lastPrice + trend);
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Gold Price',
                    data: data.prices,
                    borderColor: '#ffd700',
                    backgroundColor: 'rgba(255, 215, 0, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#ffd700',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2
                },
                {
                    label: 'High',
                    data: data.high,
                    borderColor: 'rgba(0, 212, 170, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Low',
                    data: data.low,
                    borderColor: 'rgba(255, 77, 106, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#a0a0b0',
                        font: {
                            family: "'Space Grotesk', sans-serif",
                            size: 12
                        },
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(20, 20, 30, 0.9)',
                    titleColor: '#ffd700',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 215, 0, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.raw.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#606070',
                        font: {
                            family: "'Space Grotesk', sans-serif",
                            size: 10
                        },
                        maxTicksLimit: 10
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#606070',
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 10
                        },
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
}

// Utility Functions
function formatPrice(price) {
    return price.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Add CSS animation for refresh button
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Export for global access
window.refreshAllData = refreshAllData;
