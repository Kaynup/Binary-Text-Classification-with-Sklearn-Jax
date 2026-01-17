// Configuration
const API_BASE = 'http://127.0.0.1:8000';
const API_URL = `${API_BASE}/predict`;
const MODELS_URL = `${API_BASE}/models`;
const MAX_HISTORY = 10;

// State
let history = [];
let selectedModel = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  setupEventListeners();
  updateCharCounter();
  createParticles();
  fetchModels();
});

// Fetch available models from backend
async function fetchModels() {
  const modelSelect = document.getElementById('model-select');
  const bookmarkTab = document.getElementById('bookmark-tab');
  const modelBookmark = document.getElementById('model-bookmark');

  // Setup bookmark toggle
  bookmarkTab.addEventListener('click', (e) => {
    e.stopPropagation();
    modelBookmark.classList.toggle('expanded');
  });

  // Close bookmark when clicking outside
  document.addEventListener('click', (e) => {
    if (!modelBookmark.contains(e.target)) {
      modelBookmark.classList.remove('expanded');
    }
  });

  try {
    const response = await fetch(MODELS_URL);
    if (!response.ok) throw new Error('Failed to fetch models');

    const data = await response.json();

    modelSelect.innerHTML = data.available_models.map(model =>
      `<option value="${model}" ${model === data.default ? 'selected' : ''}>${model}</option>`
    ).join('');

    selectedModel = data.default;
    updateRobotAppearance(selectedModel);

    modelSelect.addEventListener('change', (e) => {
      selectedModel = e.target.value;
      updateRobotAppearance(selectedModel);
    });

  } catch (error) {
    console.error('Error fetching models:', error);
    modelSelect.innerHTML = '<option value="sklearn-logreg">sklearn-logreg (fallback)</option>';
    selectedModel = 'sklearn-logreg';
    updateRobotAppearance(selectedModel);
  }
}

// Update robot appearance based on selected model
function updateRobotAppearance(model) {
  const robots = document.querySelectorAll('.robot');

  robots.forEach(robot => {
    // Remove existing model classes
    robot.classList.remove('jax-model', 'sklearn-model');

    // Add appropriate class based on model
    if (model.toLowerCase().includes('jax')) {
      robot.classList.add('jax-model');
    } else if (model.toLowerCase().includes('sklearn')) {
      robot.classList.add('sklearn-model');
    }
  });
}

// Event Listeners
function setupEventListeners() {
  const textInput = document.getElementById('text-input');
  textInput.addEventListener('input', updateCharCounter);
  textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      makePrediction();
    }
  });
}

// Create animated particles
function createParticles() {
  const particleContainer = document.getElementById('particles');
  const particleCount = 30;

  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.style.position = 'absolute';
    particle.style.width = Math.random() * 3 + 1 + 'px';
    particle.style.height = particle.style.width;
    particle.style.background = 'rgba(99, 102, 241, 0.3)';
    particle.style.borderRadius = '50%';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = Math.random() * 100 + '%';
    particle.style.animation = `particleFloat ${Math.random() * 20 + 10}s infinite ease-in-out`;
    particle.style.animationDelay = Math.random() * 5 + 's';
    particleContainer.appendChild(particle);
  }

  // Add particle animation to styles dynamically
  const style = document.createElement('style');
  style.textContent = `
    @keyframes particleFloat {
      0%, 100% { transform: translate(0, 0); }
      25% { transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px); }
      50% { transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px); }
      75% { transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px); }
    }
  `;
  document.head.appendChild(style);
}

// Character Counter
function updateCharCounter() {
  const textInput = document.getElementById('text-input');
  const charCount = document.getElementById('char-count');
  charCount.textContent = textInput.value.length;
}

// Main Prediction Function
async function makePrediction() {
  const textInput = document.getElementById('text-input');
  const text = textInput.value.trim();

  if (!text) {
    showMessage('Please enter some text to analyze!', 'error');
    return;
  }

  const predictBtn = document.getElementById('predict-btn');
  const loadingOverlay = document.getElementById('loading-overlay');

  // Show loading state
  predictBtn.disabled = true;
  loadingOverlay.classList.add('active');
  resetCharacter();

  const startTime = performance.now();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, model: selectedModel }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const totalTime = performance.now() - startTime;
    const networkDelay = Math.max(0, totalTime - data.inference_time_ms);

    displayResult(data, text, networkDelay);
    addToHistory(data, text, networkDelay);
    animateCharacter(data.prediction);

  } catch (error) {
    console.error('Error:', error);
    showMessage(`⚠️ Error: ${error.message}. Make sure the backend server is running!`, 'error');
    resetCharacter();
  } finally {
    predictBtn.disabled = false;
    loadingOverlay.classList.remove('active');
  }
}

// Display Result
function displayResult(data, inputText, networkDelay) {
  const resultsSection = document.getElementById('results-section');
  const resultCard = document.getElementById('result-card');
  const resultBadge = document.getElementById('result-badge');
  const resultText = document.getElementById('result-text');
  const inferenceTime = document.getElementById('inference-time');
  const networkDelayEl = document.getElementById('network-delay');
  const modelType = document.getElementById('model-type');

  const isPositive = data.prediction === 1;
  const sentiment = isPositive ? 'Positive' : 'Negative';
  const sentimentClass = isPositive ? 'positive' : 'negative';

  // Update result card
  resultCard.className = `result-card ${sentimentClass}`;
  resultBadge.className = `result-badge ${sentimentClass}`;
  resultBadge.textContent = sentiment === 'Positive' ? 'Positive' : 'Negative';

  resultText.innerHTML = `
    <strong>Input:</strong> "${inputText}"<br><br>
    <strong>Sentiment:</strong> This text expresses <strong>${sentiment.toLowerCase()}</strong> sentiment.
  `;

  inferenceTime.textContent = data.inference_time_ms;
  networkDelayEl.textContent = networkDelay.toFixed(1);
  modelType.textContent = data.model_used || (data['Jax model'] ? 'JAX Neural Network' : 'Sklearn Logistic Regression');

  resultsSection.style.display = 'block';

  // Scroll to results
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

// Animate Character
function animateCharacter(prediction) {
  const character = document.getElementById('character');
  const characterMessage = document.getElementById('character-message');
  const neutralRobot = document.getElementById('neutral-robot');
  const happyRobot = document.getElementById('happy-robot');
  const sadRobot = document.getElementById('sad-robot');

  const isPositive = prediction === 1;

  // Hide all robots first
  neutralRobot.style.display = 'none';
  happyRobot.style.display = 'none';
  sadRobot.style.display = 'none';

  if (isPositive) {
    happyRobot.style.display = 'block';
    character.className = 'character positive';
    characterMessage.textContent = "Wonderful! I detected positive vibes!";
    characterMessage.style.color = '#10b981';
  } else {
    sadRobot.style.display = 'block';
    character.className = 'character negative';
    characterMessage.textContent = "Negative sentiment detected...";
    characterMessage.style.color = '#ef4444';
  }
}

// Reset Character
function resetCharacter() {
  const character = document.getElementById('character');
  const characterMessage = document.getElementById('character-message');
  const neutralRobot = document.getElementById('neutral-robot');
  const happyRobot = document.getElementById('happy-robot');
  const sadRobot = document.getElementById('sad-robot');

  neutralRobot.style.display = 'block';
  happyRobot.style.display = 'none';
  sadRobot.style.display = 'none';

  character.className = 'character';
  characterMessage.textContent = "Ready to analyze your sentiment...";
  characterMessage.style.color = '#94a3b8';
}

// History Management
function addToHistory(data, inputText, networkDelay) {
  const historyItem = {
    text: inputText,
    prediction: data.prediction,
    inferenceTime: data.inference_time_ms,
    networkDelay: networkDelay.toFixed(1),
    timestamp: new Date().toISOString(),
    modelType: data.model_used || (data['Jax model'] ? 'JAX' : 'Sklearn')
  };

  history.unshift(historyItem);

  // Limit history size
  if (history.length > MAX_HISTORY) {
    history = history.slice(0, MAX_HISTORY);
  }

  saveHistory();
  renderHistory();
}

function renderHistory() {
  const historyList = document.getElementById('history-list');

  if (history.length === 0) {
    historyList.innerHTML = '<p class="empty-state">No analyses yet. Start by entering some text above!</p>';
    return;
  }

  historyList.innerHTML = history.map((item, index) => {
    const isPositive = item.prediction === 1;
    const sentimentClass = isPositive ? 'positive' : 'negative';
    const sentimentText = isPositive ? 'Positive' : 'Negative';
    const sentimentIcon = isPositive ? '[+]' : '[-]';
    const date = new Date(item.timestamp);
    const timeStr = date.toLocaleTimeString();
    const dateStr = date.toLocaleDateString();
    const netDelay = item.networkDelay || '--';

    return `
      <div class="history-item ${sentimentClass}">
        <div class="history-item-header">
          <span class="history-badge ${sentimentClass}">
            ${sentimentIcon} ${sentimentText}
          </span>
          <span class="history-time">${dateStr} ${timeStr}</span>
        </div>
        <p class="history-text" title="${item.text}">${item.text}</p>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 8px; display: flex; gap: 15px;">
          <span>Speed: ${item.inferenceTime} ms + ${netDelay} ms</span>
          <span>Model: ${item.modelType}</span>
        </div>
      </div>
    `;
  }).join('');
}

function clearHistory() {
  if (history.length === 0) return;

  if (confirm('Are you sure you want to clear all history?')) {
    history = [];
    saveHistory();
    renderHistory();
    showMessage('History cleared successfully!', 'success');
  }
}

function saveHistory() {
  try {
    const historyData = JSON.stringify(history);
    // Store in memory (no localStorage in Claude artifacts)
    window.sentimentHistory = historyData;
  } catch (e) {
    console.warn('Could not save history:', e);
  }
}

function loadHistory() {
  try {
    // Try to load from memory first
    if (window.sentimentHistory) {
      history = JSON.parse(window.sentimentHistory);
      renderHistory();
    }
  } catch (e) {
    console.warn('Could not load history:', e);
  }
}

// Show Message (for notifications)
function showMessage(message, type = 'info') {
  const characterMessage = document.getElementById('character-message');
  const originalMessage = characterMessage.textContent;
  const originalColor = characterMessage.style.color;

  characterMessage.textContent = message;
  characterMessage.style.color = type === 'error' ? '#ef4444' : '#10b981';

  setTimeout(() => {
    characterMessage.textContent = originalMessage;
    characterMessage.style.color = originalColor;
  }, 3000);
}