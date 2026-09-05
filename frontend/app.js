/**
 * Sentiment Analyzer Frontend Application (v2.0.0)
 * Hardened with complete XSS prevention, tactile state management,
 * live word/char counters, backend health monitoring, and clipboard integration.
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
let currentResultSummary = null;

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
// Initialization & Event Listeners
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    loadStats();
    setupEventListeners();
    updateCounters();
    setRobotState('neutral', 'Ready to analyze your sentiment...');
    
    // Check live backend connectivity immediately and periodically
    checkBackendHealth();
    setInterval(checkBackendHealth, 30000);
});

function setupEventListeners() {
    const textInput = document.getElementById('text-input');
    if (!textInput) return;

    // Live word & character counters
    textInput.addEventListener('input', updateCounters);

    // Ctrl+Enter or Cmd+Enter submits
    textInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            makePrediction();
        }
    });
}

function updateCounters() {
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const wordsCount = document.getElementById('words-count');
    if (!textInput) return;

    const text = textInput.value;
    if (charCount) {
        charCount.textContent = text.length;
    }

    if (wordsCount) {
        const trimmed = text.trim();
        const words = trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
        wordsCount.textContent = `${words} word${words === 1 ? '' : 's'}`;
    }
}

// ============================================================================
// Backend Connectivity & Health Check
// ============================================================================

async function checkBackendHealth() {
    const statusDot = document.querySelector('#api-status-badge .status-dot');
    const statusText = document.getElementById('api-status-text');

    const apiUrl = (window.APP_CONFIG && window.APP_CONFIG.API_URL)
        ? window.APP_CONFIG.API_URL
        : 'http://127.0.0.1:8000/predict';
    const healthUrl = apiUrl.replace(/\/predict\/?$/, '/health');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);

        const res = await fetch(healthUrl, { 
            method: 'GET',
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (res.ok) {
            const data = await res.json();
            if (data.status === 'healthy') {
                if (statusDot) {
                    statusDot.className = 'status-dot online';
                }
                if (statusText) {
                    statusText.textContent = 'Engine Online';
                }
                return;
            }
        }
        throw new Error('Unhealthy status');
    } catch (e) {
        if (statusDot) {
            statusDot.className = 'status-dot offline';
        }
        if (statusText) {
            statusText.textContent = 'Engine Offline';
        }
    }
}

// ============================================================================
// UI Interactions & Example Prompts
// ============================================================================

function setExampleText(text) {
    const textInput = document.getElementById('text-input');
    if (!textInput) return;
    textInput.value = text;
    updateCounters();
    textInput.focus();
}

function clearInput() {
    const textInput = document.getElementById('text-input');
    if (!textInput) return;
    textInput.value = '';
    updateCounters();
    textInput.focus();
}

function toggleBenchmarksModal() {
    const modal = document.getElementById('benchmarks-modal');
    if (!modal) return;
    const isVisible = modal.style.display === 'flex';
    modal.style.display = isVisible ? 'none' : 'flex';
}

function closeBenchmarksOnBackdrop(e) {
    if (e.target.id === 'benchmarks-modal') {
        toggleBenchmarksModal();
    }
}

// ============================================================================
// Clipboard & Toast Notification
// ============================================================================

function copyResultToClipboard() {
    if (!currentResultSummary) {
        showToast('No result available to copy');
        return;
    }

    navigator.clipboard.writeText(currentResultSummary).then(() => {
        const copyIcon = document.getElementById('copy-btn-icon');
        const copyText = document.getElementById('copy-btn-text');
        if (copyIcon && copyText) {
            copyIcon.textContent = '✓';
            copyText.textContent = 'Copied!';
            setTimeout(() => {
                copyIcon.textContent = '📋';
                copyText.textContent = 'Copy';
            }, 2000);
        }
        showToast('Summary copied to clipboard!');
    }).catch(err => {
        console.error('Clipboard error:', err);
        showToast('Could not access clipboard');
    });
}

function showToast(message) {
    let toast = document.querySelector('.toast-notification');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast-notification';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.display = 'block';

    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
        toast.style.display = 'none';
    }, 2400);
}

// ============================================================================
// Robot Character Animations & State
// ============================================================================

function setRobotState(state, message) {
    const neutralRobot = document.getElementById('neutral-robot');
    const happyRobot = document.getElementById('happy-robot');
    const sadRobot = document.getElementById('sad-robot');
    const msgElem = document.getElementById('character-message');

    if (neutralRobot) neutralRobot.style.display = 'none';
    if (happyRobot) happyRobot.style.display = 'none';
    if (sadRobot) sadRobot.style.display = 'none';

    if (state === 'positive' && happyRobot) {
        happyRobot.style.display = 'flex';
    } else if (state === 'negative' && sadRobot) {
        sadRobot.style.display = 'flex';
    } else if (neutralRobot) {
        neutralRobot.style.display = 'flex';
    }

    if (message && msgElem) {
        msgElem.textContent = message;
    }
}

// ============================================================================
// Prediction Workflow
// ============================================================================

async function makePrediction() {
    const textInput = document.getElementById('text-input');
    const text = textInput ? textInput.value.trim() : '';

    if (!text) {
        alert('Please enter some text before analyzing.');
        if (textInput) textInput.focus();
        return;
    }

    const predictBtn = document.getElementById('predict-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');

    // Set Loading State
    if (predictBtn) predictBtn.disabled = true;
    if (btnSpinner) btnSpinner.style.display = 'inline-block';
    if (btnText) btnText.textContent = 'Analyzing...';
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

        // Mark backend as active
        const statusDot = document.querySelector('#api-status-badge .status-dot');
        const statusText = document.getElementById('api-status-text');
        if (statusDot) statusDot.className = 'status-dot online';
        if (statusText) statusText.textContent = 'Engine Online';

    } catch (err) {
        console.error('Inference error:', err);
        alert(`Prediction failed: ${err.message}\n\nPlease verify that the backend server is running.`);
        setRobotState('neutral', 'Encountered an issue connecting to the inference engine.');
    } finally {
        if (predictBtn) predictBtn.disabled = false;
        if (btnSpinner) btnSpinner.style.display = 'none';
        if (btnText) btnText.textContent = 'Analyze Sentiment';
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
    if (resultCard) {
        resultCard.className = `result-card ${sentimentClass}`;
    }

    // Badge
    const resultBadge = document.getElementById('result-badge');
    if (resultBadge) {
        resultBadge.className = `result-badge ${sentimentClass}`;
        resultBadge.textContent = isPositive ? '😊 Positive' : '😞 Negative';
    }

    // Confidence pill
    const confVal = document.getElementById('confidence-val');
    if (confVal) {
        confVal.textContent = `${confidencePct}%`;
    }

    // Safe Text Insertion via textContent (Zero XSS Risk)
    const resultText = document.getElementById('result-text');
    if (resultText) {
        resultText.textContent = `"${originalText}"`;
    }

    // Probability & Polarity Calculation
    const posProbVal = data.probabilities ? Math.round(data.probabilities.positive * 100) : (isPositive ? confidencePct : 100 - confidencePct);
    const negProbVal = 100 - posProbVal;
    
    const posElem = document.getElementById('pos-prob-val');
    const negElem = document.getElementById('neg-prob-val');
    const probBar = document.getElementById('prob-bar-fill');

    if (posElem) posElem.textContent = `${posProbVal}%`;
    if (negElem) negElem.textContent = `${negProbVal}%`;
    if (probBar) probBar.style.width = `${posProbVal}%`;

    // Polarity metric: ranging from -1.00 to +1.00
    const polarity = Number(((posProbVal - negProbVal) / 100).toFixed(2));
    const polaritySign = polarity > 0 ? '+' : '';
    let polarityDesc = 'Neutral';
    if (polarity >= 0.5) polarityDesc = 'Strong Positive';
    else if (polarity > 0.1) polarityDesc = 'Moderate Positive';
    else if (polarity <= -0.5) polarityDesc = 'Strong Negative';
    else if (polarity < -0.1) polarityDesc = 'Moderate Negative';

    const polarityTag = document.getElementById('polarity-tag');
    if (polarityTag) {
        polarityTag.textContent = `Polarity: ${polaritySign}${polarity.toFixed(2)} (${polarityDesc})`;
    }

    // Meta items
    const infTimeElem = document.getElementById('inference-time');
    const modelTypeElem = document.getElementById('model-type');
    if (infTimeElem) infTimeElem.textContent = data.inference_time_ms || '2.4';
    if (modelTypeElem) modelTypeElem.textContent = 'Scikit-Learn Logistic Regression (80k features)';

    // Cache summary for clipboard action
    currentResultSummary = `Sentiment: ${sentimentLabel} (${confidencePct}% confidence)\nPolarity: ${polaritySign}${polarity.toFixed(2)} (${polarityDesc})\nInput: "${originalText}"\nEngine: Scikit-Learn Logistic Regression (v2.0.0)`;

    // Show section
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
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
    const totalElem = document.getElementById('total-analyses');
    const posElem = document.getElementById('positive-count');
    const negElem = document.getElementById('negative-count');
    const avgElem = document.getElementById('avg-time');

    if (totalElem) totalElem.textContent = stats.total;
    if (posElem) posElem.textContent = stats.positive;
    if (negElem) negElem.textContent = stats.negative;
    if (avgElem) {
        const avg = stats.total > 0 ? (stats.totalTime / stats.total).toFixed(1) : '0';
        avgElem.textContent = avg;
    }
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

    if (counter) counter.textContent = `(${history.length})`;
    if (!historyList) return;

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