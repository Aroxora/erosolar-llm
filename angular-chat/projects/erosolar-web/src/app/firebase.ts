// SPDX-License-Identifier: AGPL-3.0-only
/**
 * firebase.ts — modular Firebase initialization for the erosolar-web app.
 *
 * Initializes firebase/app and, when the environment supports it, firebase/analytics.
 * Analytics is guarded with isSupported() so SSR / unsupported browsers never throw.
 */
import { initializeApp, type FirebaseApp } from 'firebase/app';
import { getAnalytics, isSupported, logEvent, type Analytics } from 'firebase/analytics';

export const firebaseConfig = {
  apiKey: 'AIzaSyCtpuLR-rdHoC461vrQ_gActkjHGRcBTas',
  authDomain: 'erosolar-llm.firebaseapp.com',
  projectId: 'erosolar-llm',
  storageBucket: 'erosolar-llm.firebasestorage.app',
  messagingSenderId: '218680900515',
  appId: '1:218680900515:web:797c1567fade8eca597816',
  measurementId: 'G-SVMR00S44H',
};

let app: FirebaseApp | undefined;
let analytics: Analytics | undefined;

/** Initialize Firebase + (guarded) Analytics. Safe to call multiple times. */
export async function initFirebase(): Promise<FirebaseApp> {
  if (!app) {
    app = initializeApp(firebaseConfig);
  }
  if (!analytics && typeof window !== 'undefined') {
    try {
      if (await isSupported()) {
        analytics = getAnalytics(app);
      }
    } catch {
      // Analytics is best-effort; never let it break the app.
    }
  }
  return app;
}

/** Log an analytics event if (and only if) analytics initialized successfully. */
export function track(event: string, params?: { [key: string]: unknown }): void {
  if (analytics) {
    try {
      logEvent(analytics, event, params);
    } catch {
      /* swallow */
    }
  }
}
