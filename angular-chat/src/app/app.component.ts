/**
 * Erosolar - App Component (see DOMAINS.md for erosolarai.com + bundle)
 * Core app shell and layout
 *
 * Author: Bo Shang <bo@shang.software>
 */
import { Component, signal, computed, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { ChatComponent } from './components/chat/chat.component';
import { ChatService } from './services/chat.service';
import { AnalyticsService } from './services/analytics.service';
import { SessionService } from './services/session.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule, ChatComponent],
  template: `
    <div class="app-shell" [class.sidebar-open]="!sidebarCollapsed()">
      <!-- Sidebar -->
      <aside class="sidebar" [class.collapsed]="sidebarCollapsed()">
        <!-- Sidebar header -->
        <div class="sidebar-header">
          <button class="sidebar-btn" (click)="toggleSidebar()" title="Close sidebar">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <path d="M9 3v18"/>
            </svg>
          </button>

          <button class="sidebar-btn" (click)="newChat()" title="New chat">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 5v14M5 12h14"/>
            </svg>
          </button>
        </div>

        <!-- Chat list -->
        <div class="sidebar-content">
          @if (sessions().length === 0) {
            <div class="no-chats">
              <p>No previous chats</p>
            </div>
          } @else {
            <nav class="chat-list">
              @for (session of sessions(); track session.id) {
                <button
                  class="chat-item"
                  [class.active]="session.id === chatService.sessionId()"
                  (click)="loadSession(session.id)">
                  <span class="chat-title">{{ session.title }}</span>
                </button>
              }
            </nav>
          }
        </div>

        <!-- Sidebar footer - User area -->
        <div class="sidebar-footer">
          <div class="model-update-info">
            <span class="update-label">Model Updated:</span>
            <span class="update-time">{{ modelUpdateTime }}</span>
          </div>
          <button class="user-menu">
            <div class="user-avatar">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </div>
          </button>
        </div>
      </aside>

      <!-- Main area -->
      <div class="main-wrapper">
        <!-- Top header when sidebar is collapsed -->
        @if (sidebarCollapsed()) {
          <header class="top-header">
            <button class="header-btn" (click)="toggleSidebar()" title="Open sidebar">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M9 3v18"/>
              </svg>
            </button>

            <button class="header-btn" (click)="newChat()" title="New chat">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 5v14M5 12h14"/>
              </svg>
            </button>

            <div class="header-spacer"></div>

            <button class="model-dropdown">
              <span>Erosolar</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>
          </header>
        }

        <!-- Main content -->
        <main class="main-content">
          <app-chat></app-chat>
        </main>
      </div>
    </div>
  `,
  styles: [`
    .app-shell {
      display: flex;
      height: 100vh;
      width: 100vw;
      background: var(--color-bg);
      overflow: hidden;
    }

    /* Sidebar */
    .sidebar {
      width: 260px;
      min-width: 260px;
      background: var(--color-sidebar-bg);
      display: flex;
      flex-direction: column;
      transition: margin-left 0.2s ease, opacity 0.2s ease;

      &.collapsed {
        margin-left: -260px;
        opacity: 0;
        pointer-events: none;
      }
    }

    .sidebar-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px;
      height: 56px;
    }

    .sidebar-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      border-radius: 8px;
      color: var(--color-text-secondary);
      transition: background 0.15s ease, color 0.15s ease;

      &:hover {
        background: var(--color-sidebar-hover);
        color: var(--color-text);
      }
    }

    .sidebar-content {
      flex: 1;
      overflow-y: auto;
      padding: 0 8px;
    }

    .no-chats {
      padding: 16px;
      text-align: center;
      color: var(--color-text-tertiary);
      font-size: 14px;
    }

    .chat-list {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .chat-item {
      display: flex;
      align-items: center;
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      color: var(--color-text-secondary);
      font-size: 14px;
      text-align: left;
      transition: background 0.15s ease, color 0.15s ease;

      &:hover {
        background: var(--color-sidebar-hover);
        color: var(--color-text);
      }

      &.active {
        background: var(--color-sidebar-active);
        color: var(--color-text);
      }

      .chat-title {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
    }

    .sidebar-footer {
      padding: 8px;
      border-top: 1px solid var(--color-border-light);
    }

    .model-update-info {
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 8px 12px;
      margin-bottom: 4px;
      background: var(--color-surface-secondary);
      border-radius: 8px;
      font-size: 11px;

      .update-label {
        color: var(--color-text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .update-time {
        color: var(--color-text-secondary);
        font-weight: 500;
      }
    }

    .user-menu {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      color: var(--color-text);
      font-size: 14px;
      transition: background 0.15s ease;

      &:hover {
        background: var(--color-sidebar-hover);
      }

      .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--color-surface-tertiary);
        display: flex;
        align-items: center;
        justify-content: center;
      }
    }

    /* Main wrapper */
    .main-wrapper {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }

    /* Top header (when sidebar collapsed) */
    .top-header {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 8px 12px;
      height: 56px;
    }

    .header-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      border-radius: 8px;
      color: var(--color-text-secondary);
      transition: background 0.15s ease, color 0.15s ease;

      &:hover {
        background: var(--color-surface-secondary);
        color: var(--color-text);
      }
    }

    .header-spacer {
      flex: 1;
    }

    .model-dropdown {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 8px 12px;
      border-radius: 8px;
      color: var(--color-text);
      font-size: 16px;
      font-weight: 600;
      transition: background 0.15s ease;

      &:hover {
        background: var(--color-surface-secondary);
      }

      svg {
        color: var(--color-text-secondary);
      }
    }

    /* Main content */
    .main-content {
      flex: 1;
      overflow: hidden;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .sidebar {
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        z-index: 1000;
        box-shadow: 4px 0 16px rgba(0, 0, 0, 0.4);

        &.collapsed {
          margin-left: -260px;
        }
      }
    }
  `]
})
export class AppComponent implements OnInit {
  // Inject session service directly for signal-based sessions (pure Angular)
  private sessionService = inject(SessionService);

  // Sessions via Angular computed signal - automatically updates when sessionService.sessions changes
  sessions = computed(() => this.sessionService.sessions());
  sidebarCollapsed = signal(false);

  // Model last update timestamp - fetched from version.json
  modelUpdateTime = 'Loading...';

  constructor(
    public chatService: ChatService,
    private analyticsService: AnalyticsService,
    private http: HttpClient
  ) {
    // Load sessions on startup
    this.initSessions();
    void this.analyticsService.logEvent('app_loaded');
  }

  ngOnInit(): void {
    this.loadModelVersion();
  }

  private loadModelVersion(): void {
    this.http.get<any>('/assets/version.json').subscribe({
      next: (data) => {
        if (data?.modelUpdateTime) {
          const date = new Date(data.modelUpdateTime);
          this.modelUpdateTime = date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true,
            timeZoneName: 'short'
          });
        }
      },
      error: () => {
        this.modelUpdateTime = 'Unknown';
      }
    });
  }

  private async initSessions(): Promise<void> {
    // Small delay to let Firebase initialize, then load sessions
    setTimeout(() => this.sessionService.loadUserSessions(), 500);
  }

  async newChat(): Promise<void> {
    await this.chatService.newSession();
    // Sessions update automatically via signals - no need to manually reload
  }

  loadSession(id: string): void {
    this.chatService.loadSessionById(id);
  }

  toggleSidebar(): void {
    this.sidebarCollapsed.update(v => !v);
  }
}
