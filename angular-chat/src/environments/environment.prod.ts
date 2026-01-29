/**
 * Erosolar Chat Production Environment
 * Author: Bo Shang <bo@shang.software>
 */
export const environment = {
  production: true,
  // Cloud Run API endpoint
  apiUrl: 'https://erosolar-837987143968.us-central1.run.app',
  chatCompletionsPath: '/v1/responses',
  searchPath: '/search',
  modelName: 'erosolar',
  tavilyApiKey: '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',
  // Firebase config
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
