// Runtime Configuration for Sentiment Analyzer (v2.0.0)
// =======================================================
// Connected to live Railway backend production endpoint.

window.APP_CONFIG = {
    // Railway backend endpoint
    API_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://127.0.0.1:8000/predict'
        : 'https://binary-text-classification-with-sklearn-jax-production.up.railway.app/predict',

    APP_NAME: 'Sentiment Analyzer',
    VERSION: '2.0.0',
    FRAMEWORK: 'Scikit-Learn Pure'
};

console.log('App Configuration loaded:', window.APP_CONFIG);
