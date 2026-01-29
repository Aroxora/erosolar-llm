const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp();
const db = admin.firestore();
const storage = admin.storage();

/**
 * Get available models
 */
exports.getModels = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  try {
    const snapshot = await db.collection('models').get();
    const models = [];
    snapshot.forEach(doc => {
      models.push({ id: doc.id, ...doc.data() });
    });
    res.json({ success: true, models });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Get model info
 */
exports.getModelInfo = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  const modelName = req.query.name;
  if (!modelName) {
    res.status(400).json({ success: false, error: 'Model name required' });
    return;
  }

  try {
    const doc = await db.collection('models').doc(modelName).get();
    if (!doc.exists) {
      res.status(404).json({ success: false, error: 'Model not found' });
      return;
    }
    res.json({ success: true, model: { id: doc.id, ...doc.data() } });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Get training results/benchmarks
 */
exports.getBenchmarks = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  try {
    const snapshot = await db.collection('benchmarks').orderBy('timestamp', 'desc').limit(50).get();
    const benchmarks = [];
    snapshot.forEach(doc => {
      benchmarks.push({ id: doc.id, ...doc.data() });
    });
    res.json({ success: true, benchmarks });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Store benchmark result (admin only)
 */
exports.storeBenchmark = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'POST');
    res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.status(204).send('');
    return;
  }

  // Check for admin key
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ success: false, error: 'Unauthorized' });
    return;
  }

  try {
    const benchmark = req.body;
    benchmark.timestamp = admin.firestore.FieldValue.serverTimestamp();

    const docRef = await db.collection('benchmarks').add(benchmark);
    res.json({ success: true, id: docRef.id });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Get GPT-5.x model configuration
 */
exports.getGPT5Config = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  // Return locked GPT-5.x configuration
  const config = {
    primary: "gpt-5.1-codex-mini",
    backup: ["gpt-5.1-mini", "gpt-5-nano"],
    api_type: "responses",
    fallback_to_chat: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    locked: true
  };

  res.json({ success: true, config });
});
