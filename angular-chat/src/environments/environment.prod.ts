/**
 * Erosolar Chat Production Environment
 * Author: Bo Shang <bo@shang.software>
 */
export const environment = {
  production: true,
  // Cloud Run API endpoint (update to custom domain after acquiring erosoralai.com per DOMAINS.md)
  apiUrl: 'https://erosolar-837987143968.us-central1.run.app',
  chatCompletionsPath: '/v1/responses',
  searchPath: '/search',
  modelName: 'erosolar',
  tavilyApiKey: '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',
  // Firebase config
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
