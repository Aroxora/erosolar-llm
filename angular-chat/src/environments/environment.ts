/**
 * Erosolar Chat Environment Configuration
 * Author: Bo Shang <bo@shang.software>
 */
export const environment = {
  production: false,
  // Cloud Run API endpoint (change to your deployed URL)
  apiUrl: 'http://localhost:8080',
  // Response API endpoint path
  chatCompletionsPath: '/v1/responses',
  // Tavily search endpoint
  searchPath: '/search',
  // Model name
  modelName: 'erosolar',
  // Tavily API key (for client-side search)
  tavilyApiKey: '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',
  // Firebase config (same project for dev)
  firebase: {
    apiKey: 'AIzaSyCtpuLR-rdHoC461vrQ_gActkjHGRcBTas',
    authDomain: 'erosolar-llm.firebaseapp.com',
    projectId: 'erosolar-llm',
    storageBucket: 'erosolar-llm.firebasestorage.app',
    messagingSenderId: '218680900515',
    appId: '1:218680900515:web:797c1567fade8eca597816',
    measurementId: 'G-SVMR00S44H'
  }
};
