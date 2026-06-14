/**
 * Erosolar Session Service
 * Handles session persistence with Firestore
 *
 * Architecture:
 * - Anonymous users: Sessions stored in public_sessions collection
 * - Authenticated users: Sessions stored under users/{uid}/sessions
 *
 * Author: Bo Shang <bo@shang.software>
 */
import { Injectable, signal, computed } from '@angular/core';
import { FirebaseApp } from 'firebase/app';
import {
  getFirestore,
  Firestore,
  collection,
  doc,
  setDoc,
  getDoc,
  getDocs,
  deleteDoc,
  updateDoc,
  serverTimestamp,
  query,
  orderBy,
  limit,
  Timestamp
} from 'firebase/firestore';
import {
  getAuth,
  Auth,
  User,
  onAuthStateChanged
} from 'firebase/auth';
import { AnalyticsService } from './analytics.service';
import { getFirebaseApp } from './firebase-app';

export interface SessionMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  thinking?: string;
  answer?: string;
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: SessionMessage[];
  createdAt: Date;
  updatedAt: Date;
}

@Injectable({
  providedIn: 'root'
})
export class SessionService {
  private app: FirebaseApp;
  private db: Firestore;
  private auth: Auth;

  // Angular signals-based state (pure Angular reactivity)
  private currentUserSignal = signal<User | null>(null);
  private currentSessionIdSignal = signal<string | null>(null);
  private sessionsSignal = signal<ChatSession[]>([]);
  private isLoadingSignal = signal<boolean>(false);

  // Public computed signals
  public currentUser = computed(() => this.currentUserSignal());
  public currentSessionId = computed(() => this.currentSessionIdSignal());
  public sessions = computed(() => this.sessionsSignal());
  public isLoading = computed(() => this.isLoadingSignal());
  public isAuthenticated = computed(() => this.currentUserSignal() !== null && !this.currentUserSignal()?.isAnonymous);

  // Storage keys
  private readonly SESSION_ID_KEY = 'erosolar_session_id';

  constructor(private analyticsService: AnalyticsService) {
    // Initialize Firebase
    this.app = getFirebaseApp();
    this.db = getFirestore(this.app);
    this.auth = getAuth(this.app);

    // Listen for auth state changes
    onAuthStateChanged(this.auth, (user) => {
      this.currentUserSignal.set(user);
      void this.analyticsService.setUser(user);
      if (user) {
        this.loadUserSessions();
      }
    });

    // Initialize session
    this.initSession();
  }

  /**
   * Initialize session - create new or load existing
   */
  private async initSession(): Promise<void> {
    // Check for existing session ID in localStorage
    let sessionId = localStorage.getItem(this.SESSION_ID_KEY);

    if (!sessionId) {
      // Generate new session ID
      sessionId = this.generateSessionId();
      localStorage.setItem(this.SESSION_ID_KEY, sessionId);
    }

    this.currentSessionIdSignal.set(sessionId);

    // Try to load existing session
    await this.loadSession(sessionId);
  }

  /**
   * Generate a unique session ID
   */
  private generateSessionId(): string {
    const timestamp = Date.now().toString(36);
    const randomPart = Math.random().toString(36).substring(2, 11);
    return `session_${timestamp}_${randomPart}`;
  }

  /**
   * Create a new chat session
   * Note: Session is only persisted to Firestore when first message is sent
   */
  async newSession(): Promise<string> {
    const sessionId = this.generateSessionId();
    localStorage.setItem(this.SESSION_ID_KEY, sessionId);
    this.currentSessionIdSignal.set(sessionId);

    // Don't save to Firestore yet - wait for first message
    // This prevents empty "New Chat" entries from cluttering the sidebar

    return sessionId;
  }

  /**
   * Load session from Firestore
   */
  async loadSession(sessionId: string): Promise<ChatSession | null> {
    this.isLoadingSignal.set(true);

    try {
      const user = this.currentUserSignal();
      let docRef;

      if (user && !user.isAnonymous) {
        // Authenticated user - load from user's sessions
        docRef = doc(this.db, 'users', user.uid, 'sessions', sessionId);
      } else {
        // Anonymous - load from public_sessions
        docRef = doc(this.db, 'public_sessions', sessionId);
      }

      const docSnap = await getDoc(docRef);

      if (docSnap.exists()) {
        const data = docSnap.data();
        const session: ChatSession = {
          id: sessionId,
          title: data['title'] || 'Chat',
          messages: (data['messages'] || []).map((m: any) => ({
            ...m,
            timestamp: m.timestamp?.toDate ? m.timestamp.toDate() : new Date(m.timestamp)
          })),
          createdAt: data['createdAt']?.toDate ? data['createdAt'].toDate() : new Date(),
          updatedAt: data['updatedAt']?.toDate ? data['updatedAt'].toDate() : new Date()
        };
        return session;
      }

      return null;
    } catch (error) {
      console.error('Error loading session:', error);
      return null;
    } finally {
      this.isLoadingSignal.set(false);
    }
  }

  /**
   * Save session to Firestore
   */
  async saveSessionToFirestore(session: ChatSession): Promise<void> {
    try {
      const user = this.currentUserSignal();
      let docRef;

      if (user && !user.isAnonymous) {
        // Authenticated user - save to user's sessions
        docRef = doc(this.db, 'users', user.uid, 'sessions', session.id);
      } else {
        // Anonymous - save to public_sessions
        docRef = doc(this.db, 'public_sessions', session.id);
      }

      const sessionData = {
        title: session.title,
        messages: session.messages.map(m => ({
          role: m.role,
          content: m.content,
          thinking: m.thinking || '',
          answer: m.answer || '',
          timestamp: Timestamp.fromDate(m.timestamp)
        })),
        createdAt: Timestamp.fromDate(session.createdAt),
        updatedAt: serverTimestamp()
      };

      await setDoc(docRef, sessionData, { merge: true });
    } catch (error) {
      console.error('Error saving session:', error);
    }
  }

  /**
   * Save messages to current session
   */
  async saveMessages(messages: SessionMessage[]): Promise<void> {
    const sessionId = this.currentSessionIdSignal();
    if (!sessionId) return;

    const session: ChatSession = {
      id: sessionId,
      title: this.generateTitle(messages),
      messages,
      createdAt: new Date(), // Will be preserved by merge
      updatedAt: new Date()
    };

    await this.saveSessionToFirestore(session);

    // Update local sessions list immediately for sidebar display
    this.sessionsSignal.update(sessions => {
      const existing = sessions.findIndex(s => s.id === sessionId);
      if (existing >= 0) {
        // Update existing and move to top
        const updated = [...sessions];
        updated.splice(existing, 1);
        return [session, ...updated];
      } else {
        // Add new session at top
        return [session, ...sessions];
      }
    });
  }

  /**
   * Generate session title from first message
   */
  private generateTitle(messages: SessionMessage[]): string {
    const firstUserMessage = messages.find(m => m.role === 'user');
    if (firstUserMessage) {
      const content = firstUserMessage.content;
      return content.length > 30 ? content.substring(0, 30) + '...' : content;
    }
    return 'New Chat';
  }

  /**
   * Load all sessions for user (authenticated or anonymous)
   */
  async loadUserSessions(): Promise<void> {
    this.isLoadingSignal.set(true);

    try {
      const user = this.currentUserSignal();
      let sessionsRef;

      if (user && !user.isAnonymous) {
        // Authenticated user
        sessionsRef = collection(this.db, 'users', user.uid, 'sessions');
      } else {
        // Anonymous user - load from public_sessions
        sessionsRef = collection(this.db, 'public_sessions');
      }

      const q = query(sessionsRef, orderBy('updatedAt', 'desc'), limit(50));
      const snapshot = await getDocs(q);

      const sessions: ChatSession[] = snapshot.docs
        .map(doc => {
          const data = doc.data();
          return {
            id: doc.id,
            title: data['title'] || 'Chat',
            messages: data['messages'] || [],
            createdAt: data['createdAt']?.toDate() || new Date(),
            updatedAt: data['updatedAt']?.toDate() || new Date()
          };
        })
        // Filter out empty sessions with no real title
        .filter(s => s.title !== 'New Chat' || s.messages.length > 0);

      this.sessionsSignal.set(sessions);
    } catch (error) {
      console.error('Error loading user sessions:', error);
    } finally {
      this.isLoadingSignal.set(false);
    }
  }

  /**
   * Delete a session
   */
  async deleteSession(sessionId: string): Promise<void> {
    try {
      const user = this.currentUserSignal();
      let docRef;

      if (user && !user.isAnonymous) {
        docRef = doc(this.db, 'users', user.uid, 'sessions', sessionId);
      } else {
        docRef = doc(this.db, 'public_sessions', sessionId);
      }

      await deleteDoc(docRef);

      // Update local state
      this.sessionsSignal.update(sessions =>
        sessions.filter(s => s.id !== sessionId)
      );

      // If deleted current session, create new one
      if (sessionId === this.currentSessionIdSignal()) {
        await this.newSession();
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  }

  /**
   * Switch to a different session
   */
  async switchSession(sessionId: string): Promise<void> {
    localStorage.setItem(this.SESSION_ID_KEY, sessionId);
    this.currentSessionIdSignal.set(sessionId);
    await this.loadSession(sessionId);
  }

  /**
   * Get current session ID
   */
  getSessionId(): string {
    return this.currentSessionIdSignal() || '';
  }
}
