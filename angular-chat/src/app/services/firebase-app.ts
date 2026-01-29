import { FirebaseApp, getApp, getApps, initializeApp } from 'firebase/app';
import { environment } from '../../environments/environment';

export const getFirebaseApp = (): FirebaseApp => {
  if (getApps().length) {
    return getApp();
  }

  return initializeApp(environment.firebase);
};
