/**
 * Sentiment Analyzer Frontend Application (v2.0.0)
 * Hardened with complete XSS prevention, tactile state management,
 * and robust error handling.
 */

// Application State
let history = [];
let stats = {
    total: 0,
    positive: 0,
    negative: 0,
    totalTime: 0
};
const MAX_HISTORY = 30;

// ============================================================================
// Security & XSS Prevention
// ============================================================================

/**
 * Strictly sanitizes any string before rendering into DOM templates.
 * Neutralizes <script>, <img> onerror, attribute injections, and html entities.
 */
function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    loadStats();
    setupEventListeners();
    setRobotState('neutral', 'Ready to analyze your sentiment...');
});

function setupEventListeners() {
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');

    // Live character counter
    textInput.addEventListener('input', () => {
        charCount.textContent = textInput.value.length;
    });

    // Ctrl+Enter or Cmd+Enter submits
    textInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            makePrediction();
        }
    });
}

// ============================================================================
// UI Interactions & Example Prompts
// ============================================================================

function setExampleText(text) {
    const textInput = document.getElementById('text-input');
    textInput.value = text;
    document.getElementById('char-count').textContent = text.length;
    textInput.focus();
}

function clearInput() {
    const textInput = document.getElementById('text-input');
    textInput.value = '';
    document.getElementById('char-count').textContent = '0';
    textInput.focus();
}

function toggleBenchmarksModal() {
    const modal = document.getElementById('benchmarks-modal');
    const isVisible = modal.style.display === 'flex';
    modal.style.display = isVisible ? 'none' : 'flex';
}

function closeBenchmarksOnBackdrop(e) {
    if (e.target.id === 'benchmarks-modal') {
        toggleBenchmarksModal();
    }
}

// ============================================================================
// Robot Character Animations & State
// ============================================================================

function setRobotState(state, message) {
    const neutralRobot = document.getElementById('neutral-robot');
    const happyRobot = document.getElementById('happy-robot');
    const sadRobot = document.getElementById('sad-robot');
    const msgElem = document.getElementById('character-message');

    // Reset visibility
    neutralRobot.style.display = 'none';
    happyRobot.style.display = 'none';
    sadRobot.style.display = 'none';

    if (state === 'positive') {
        happyRobot.style.display = 'flex';
    } else if (state === 'negative') {
        sadRobot.style.display = 'flex';
    } else {
        neutralRobot.style.display = 'flex';
    }

    if (message) {
        msgElem.textContent = message;
    }
}

// ============================================================================
// Prediction Workflow
// ============================================================================

async function makePrediction() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();

    if (!text) {
        alert('Please enter some text before analyzing.');
        textInput.focus();
        return;
    }

    const predictBtn = document.getElementById('predict-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');

    // Set Loading State
    predictBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = 'Analyzing...';
    setRobotState('neutral', 'Processing sentence through TF-IDF vectors...');

    // Resolve API URL
    const apiUrl = (window.APP_CONFIG && window.APP_CONFIG.API_URL)
        ? window.APP_CONFIG.API_URL
        : 'http://127.0.0.1:8000/predict';

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                model: 'sklearn-logreg'
            })
        });

        if (response.status === 429) {
            const errorData = await response.json().catch(() => ({}));
            const retryAfter = errorData.retry_after || 60;
            alert(`Rate limit reached. Please wait ${retryAfter} seconds before trying again.`);
            setRobotState('neutral', 'Rate limit active. Pausing requests...');
            return;
        }

        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            throw new Error(errBody.message || `Server error (HTTP ${response.status})`);
        }

        const data = await response.json();
        displayResults(data, text);
        updateHistory(data, text);
        updateStats(data);

    } catch (err) {
        console.error('Inference error:', err);
        alert(`Prediction failed: ${err.message}\n\nPlease check that the backend is running.`);
        setRobotState('neutral', 'Encountered an issue connecting to the inference engine.');
    } finally {
        predictBtn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.textContent = 'Analyze Sentiment';
    }
}

// ============================================================================
// Result Rendering (XSS-Safe)
// ============================================================================

function displayResults(data, originalText) {
    const isPositive = data.prediction === 1;
    const sentimentLabel = isPositive ? 'Positive' : 'Negative';
    const sentimentClass = isPositive ? 'positive' : 'negative';
    const confidencePct = Math.round((data.confidence || 0.85) * 100);

    // Robot reaction & speech
    if (isPositive) {
        const positivePhrases = [
            "Splendid! That feels noticeably positive! 💚",
            "Delightful sentiment detected! Radiant energy.",
            "That puts a warm smile on my circuit face!"
        ];
        const phrase = positivePhrases[Math.floor(Math.random() * positivePhrases.length)];
        setRobotState('positive', phrase);
    } else {
        const negativePhrases = [
            "Oh dear... That carries a downcast sentiment. 💙",
            "Detected frustration or dissatisfaction in that statement.",
            "Sending sympathy... Sentiment registers as negative."
        ];
        const phrase = negativePhrases[Math.floor(Math.random() * negativePhrases.length)];
        setRobotState('negative', phrase);
    }

    // Card state
    const resultCard = document.getElementById('result-card');
    resultCard.className = `result-card ${sentimentClass}`;

    // Badge
    const resultBadge = document.getElementById('result-badge');
    resultBadge.className = `result-badge ${sentimentClass}`;
    resultBadge.textContent = isPositive ? '😊 Positive' : '😞 Negative';

    // Confidence pill
    document.getElementById('confidence-val').textContent = `${confidencePct}%`;

    // Safe Text Insertion via textContent (Zero XSS Risk)
    const resultText = document.getElementById('result-text');
    resultText.textContent = `"${originalText}"`;

    // Probability Track
    const posProb = data.probabilities ? Math.round(data.probabilities.positive * 100) : (isPositive ? confidencePct : 100 - confidencePct);
    const negProb = 100 - posProb;
    document.getElementById('pos-prob-val').textContent = `${posProb}%`;
    document.getElementById('neg-prob-val').textContent = `${negProb}%`;
    document.getElementById('prob-bar-fill').style.width = `${posProb}%`;

    // Meta items
    document.getElementById('inference-time').textContent = data.inference_time_ms || '2.4';
    document.getElementById('model-type').textContent = 'Scikit-Learn Logistic Regression (80k features)';

    // Show section
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// History & Statistics
// ============================================================================

function updateStats(data) {
    const isPositive = data.prediction === 1;
    stats.total += 1;
    if (isPositive) {
        stats.positive += 1;
    } else {
        stats.negative += 1;
    }
    stats.totalTime += Number(data.inference_time_ms || 2.5);

    saveStats();
    renderStats();
}

function renderStats() {
    document.getElementById('total-analyses').textContent = stats.total;
    document.getElementById('positive-count').textContent = stats.positive;
    document.getElementById('negative-count').textContent = stats.negative;
    const avg = stats.total > 0 ? (stats.totalTime / stats.total).toFixed(1) : '0';
    document.getElementById('avg-time').textContent = avg;
}

function updateHistory(data, originalText) {
    const item = {
        id: Date.now(),
        text: originalText,
        prediction: data.prediction,
        confidence: data.confidence,
        inference_time_ms: data.inference_time_ms,
        timestamp: new Date().toISOString()
    };

    history.unshift(item);
    if (history.length > MAX_HISTORY) {
        history = history.slice(0, MAX_HISTORY);
    }

    saveHistory();
    renderHistory();
}

function renderHistory() {
    const historyList = document.getElementById('history-list');
    const counter = document.getElementById('history-counter');

    counter.textContent = `(${history.length})`;

    if (history.length === 0) {
        historyList.innerHTML = '<p class="empty-state">No analyses yet. Enter text above to inspect sentiment.</p>';
        return;
    }

    // Build DOM elements safely without innerHTML string interpolation of raw text
    historyList.innerHTML = '';

    history.forEach(item => {
        const isPositive = item.prediction === 1;
        const card = document.createElement('div');
        card.className = `history-item ${isPositive ? 'positive' : 'negative'}`;

        const header = document.createElement('div');
        header.className = 'history-item-header';

        const badge = document.createElement('span');
        badge.className = `history-badge ${isPositive ? 'positive' : 'negative'}`;
        badge.textContent = isPositive ? '😊 Positive' : '😞 Negative';

        const timeSpan = document.createElement('span');
        timeSpan.className = 'history-time';
        const dateObj = new Date(item.timestamp);
        timeSpan.textContent = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        header.appendChild(badge);
        header.appendChild(timeSpan);

        // Text rendered via textContent -> Guaranteed 100% XSS immune
        const textP = document.createElement('p');
        textP.className = 'history-text';
        textP.textContent = item.text;
        textP.title = item.text;

        const metaDiv = document.createElement('div');
        metaDiv.className = 'history-meta';
        metaDiv.innerHTML = `<span>⚡ ${escapeHtml(String(item.inference_time_ms || '--'))} ms</span><span>🎯 ${Math.round((item.confidence || 0.85) * 100)}% conf</span>`;

        card.appendChild(header);
        card.appendChild(textP);
        card.appendChild(metaDiv);

        historyList.appendChild(card);
    });
}

function clearHistory() {
    if (history.length === 0) return;
    if (confirm('Clear all recent analysis history?')) {
        history = [];
        saveHistory();
        renderHistory();
    }
}

// LocalStorage Helpers
function saveHistory() {
    try {
        localStorage.setItem('sentiment_analyzer_history_v2', JSON.stringify(history));
    } catch (e) {
        console.warn('Could not save history to localStorage', e);
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('sentiment_analyzer_history_v2');
        if (saved) {
            history = JSON.parse(saved);
            renderHistory();
        }
    } catch (e) {
        history = [];
    }
}

function saveStats() {
    try {
        localStorage.setItem('sentiment_analyzer_stats_v2', JSON.stringify(stats));
    } catch (e) {
        console.warn('Could not save stats to localStorage', e);
    }
}

function loadStats() {
    try {
        const saved = localStorage.getItem('sentiment_analyzer_stats_v2');
        if (saved) {
            stats = JSON.parse(saved);
            renderStats();
        }
    } catch (e) {
        stats = { total: 0, positive: 0, negative: 0, totalTime: 0 };
    }
}
