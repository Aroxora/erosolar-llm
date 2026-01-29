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
    apiKey: 'AIzaSyChdFlFhOTuZv4q5CEAYR5smEzHHTRz5qg',
    authDomain: 'america-is-finally-back.firebaseapp.com',
    projectId: 'america-is-finally-back',
    storageBucket: 'america-is-finally-back.firebasestorage.app',
    messagingSenderId: '13762901352',
    appId: '1:13762901352:web:erosolar',
    measurementId: 'G-M8C274Q0KT'
  }
};
