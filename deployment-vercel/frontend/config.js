// Runtime Configuration for Sentiment Analyzer
// ================================================
// Update this file with your deployed backend URL before deploying to Vercel
//
// For local development: Use http://127.0.0.1:8000/predict
// For production: Use your Railway/Render URL, e.g., https://your-app.railway.app/predict

window.APP_CONFIG = {
    // ✅ Backend deployed on Railway
    API_URL: 'https://lavish-generosity-production.up.railway.app/predict',

    // App metadata
    APP_NAME: 'Sentiment Analyzer',
    VERSION: '1.0.0'
};

// Log configuration on load
console.log('📋 App Configuration loaded:', window.APP_CONFIG);
