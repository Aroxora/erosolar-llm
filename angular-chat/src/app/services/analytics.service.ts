import { Injectable } from '@angular/core';
import { Analytics, getAnalytics, isSupported, logEvent, setUserId, setUserProperties } from 'firebase/analytics';
import type { User } from 'firebase/auth';
import { environment } from '../../environments/environment';
import { getFirebaseApp } from './firebase-app';

@Injectable({
  providedIn: 'root'
})
export class AnalyticsService {
  private analytics: Analytics | null = null;
  private initPromise: Promise<void>;

  constructor() {
    this.initPromise = this.init();
  }

  private async init(): Promise<void> {
    if (!environment.firebase?.measurementId) {
      return;
    }

    try {
      const supported = await isSupported();
      if (!supported) {
        return;
      }

      const app = getFirebaseApp();
      this.analytics = getAnalytics(app);
    } catch (error) {
      return;
    }
  }

  private async isReady(): Promise<boolean> {
    await this.initPromise;
    return this.analytics !== null;
  }

  async logEvent(name: string, params?: Record<string, string | number>): Promise<void> {
    if (!(await this.isReady())) {
      return;
    }

    logEvent(this.analytics as Analytics, name, params);
  }

  async setUser(user: User | null): Promise<void> {
    if (!(await this.isReady())) {
      return;
    }

    if (user?.uid && !user.isAnonymous) {
      setUserId(this.analytics as Analytics, user.uid);
    }

    setUserProperties(this.analytics as Analytics, {
      auth_state: user ? (user.isAnonymous ? 'anonymous' : 'authenticated') : 'none'
    });
  }
}
