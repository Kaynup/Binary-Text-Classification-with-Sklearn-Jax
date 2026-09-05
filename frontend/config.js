// Runtime Configuration for Sentiment Analyzer (v2.0.0)
// =======================================================
// Update API_URL with your deployed Railway backend URL before deploying to Vercel.
//
// Local development: http://127.0.0.1:8000/predict
// Production: https://your-railway-app.up.railway.app/predict

window.APP_CONFIG = {
    // Railway backend endpoint
    API_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://127.0.0.1:8000/predict'
        : 'https://lavish-generosity-production.up.railway.app/predict',

    APP_NAME: 'Sentiment Analyzer',
    VERSION: '2.0.0',
    FRAMEWORK: 'Scikit-Learn Pure'
};

console.log('App Configuration loaded:', window.APP_CONFIG);
