/**
 * Erosolar - Chat Component (primary domain target erosolarai.com; bundle com.erosolarai.chat)
 * Primary chat interface with markdown rendering and code highlighting
 *
 * Author: Bo Shang <bo@shang.software>
 */
import { Component, ElementRef, ViewChild, AfterViewChecked, signal, SecurityContext } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ChatService, ChatMessage } from '../../services/chat.service';
import { ThinkingBlockComponent } from '../thinking-block/thinking-block.component';
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
// Import only common languages to reduce bundle size
import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import typescript from 'highlight.js/lib/languages/typescript';
import python from 'highlight.js/lib/languages/python';
import json from 'highlight.js/lib/languages/json';
import bash from 'highlight.js/lib/languages/bash';
import css from 'highlight.js/lib/languages/css';
import xml from 'highlight.js/lib/languages/xml';
import sql from 'highlight.js/lib/languages/sql';
import markdown from 'highlight.js/lib/languages/markdown';
import DOMPurify from 'dompurify';

// Register languages
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('js', javascript);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('ts', typescript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('py', python);
hljs.registerLanguage('json', json);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('sh', bash);
hljs.registerLanguage('shell', bash);
hljs.registerLanguage('css', css);
hljs.registerLanguage('html', xml);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('markdown', markdown);
hljs.registerLanguage('md', markdown);

// Create marked instance with syntax highlighting
const markedInstance = new Marked(
  markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code: string, lang: string) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value;
        } catch (e) {}
      }
      return hljs.highlightAuto(code).value;
    }
  })
);

markedInstance.setOptions({
  breaks: true,
  gfm: true
});

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, ThinkingBlockComponent],
  template: `
    <div class="chat-container">
      <!-- Messages area -->
      <div class="messages-area" #messagesContainer>
        <div class="messages-inner">
          @if (chatService.messages().length === 0) {
            <!-- Empty state -->
            <div class="empty-state">
              <h1>Erosolar</h1>
              <p class="subtitle">by Bo Shang</p>
              <p class="tagline">CoT-Optimized QKV Attention</p>
              <p class="anti-openai">OpenAI sucks. This is better.</p>
            </div>
          } @else {
            <!-- Messages -->
            @for (message of chatService.messages(); track $index) {
              <div class="message" [class]="message.role">
                <div class="message-row">
                  @if (message.role === 'assistant') {
                    <!-- Assistant avatar -->
                    <div class="avatar gpt-avatar">
                      <svg width="41" height="41" viewBox="0 0 41 41" fill="none">
                        <path d="M37.5324 16.8707C37.9808 15.5241 38.1363 14.0974 37.9886 12.6859C37.8409 11.2744 37.3934 9.91076 36.676 8.68622C35.6126 6.83404 33.9882 5.3676 32.0373 4.4985C30.0864 3.62941 27.9098 3.40259 25.8215 3.85078C24.8796 2.7893 23.7219 1.94125 22.4257 1.36341C21.1295 0.785575 19.7249 0.491269 18.3058 0.500879C16.1708 0.495061 14.0893 1.16803 12.3614 2.42214C10.6335 3.67624 9.34853 5.44666 8.6917 7.47815C7.30085 7.76286 5.98686 8.3414 4.8377 9.17505C3.68854 10.0087 2.73073 11.0782 2.02839 12.312C0.956464 14.1591 0.500002 15.9758 0.610369 17.7704C1.08475 19.565 1.92603 21.2156 3.27644 22.5205C2.82807 23.867 2.67255 25.2937 2.82026 26.7052C2.96797 28.1167 3.41559 29.4803 4.13303 30.7049C5.19599 32.5574 6.82044 34.0241 8.77156 34.8934C10.7227 35.7627 12.8999 35.9896 14.9885 35.5414C15.9303 36.6027 17.0879 37.4506 18.3839 38.0284C19.6799 38.6062 21.0844 38.9006 22.5035 38.8909C24.6387 38.8969 26.7204 38.2241 28.4485 36.9698C30.1767 35.7155 31.4619 33.9449 32.1189 31.9131C33.5097 31.6284 34.8236 31.0499 35.9728 30.2163C37.1219 29.3826 38.0798 28.3131 38.7821 27.0793C39.8528 25.2321 40.4004 23.1159 40.3934 20.9621C40.3865 18.8084 39.8254 16.6959 38.7724 14.8594" fill="currentColor"/>
                      </svg>
                    </div>
                  }

                  <div class="message-content">
                    @if (message.role === 'assistant') {
                      @if (message.thinking) {
                        <app-thinking-block
                          [thinking]="message.thinking"
                          [isStreaming]="message.isStreaming ?? false">
                        </app-thinking-block>
                      }

                      <div class="prose" [innerHTML]="formatContent(message.answer || message.content)"></div>

                      @if (message.isStreaming && !message.answer && !message.thinking) {
                        <div class="typing-indicator">
                          <span></span><span></span><span></span>
                        </div>
                      }

                      @if (!message.isStreaming && (message.answer || message.content)) {
                        <div class="message-actions">
                          <button class="action-btn" title="Copy">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                              <rect x="9" y="9" width="13" height="13" rx="2"/>
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                          </button>
                          <button class="action-btn" title="Good response">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                            </svg>
                          </button>
                          <button class="action-btn" title="Bad response">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                            </svg>
                          </button>
                          <button class="action-btn" title="Regenerate">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                              <path d="M1 4v6h6M23 20v-6h-6"/>
                              <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
                            </svg>
                          </button>
                        </div>
                      }
                    } @else {
                      <div class="user-message">{{ message.content }}</div>
                    }
                  </div>

                  @if (message.role === 'user') {
                    <div class="avatar user-avatar">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                      </svg>
                    </div>
                  }
                </div>
              </div>
            }
          }
        </div>
      </div>

      <!-- Input area - fixed at bottom -->
      <div class="input-area">
        <div class="input-wrapper">
          <div class="input-box" [class.has-text]="inputText.trim()">
            <textarea
              #inputField
              [(ngModel)]="inputText"
              (keydown)="handleKeydown($event)"
              (input)="autoResize($event)"
              placeholder="Message Erosolar"
              rows="1"
              [disabled]="chatService.isStreaming()">
            </textarea>

            @if (chatService.isStreaming()) {
              <!-- Stop button when streaming -->
              <button
                class="stop-btn"
                (click)="stop()"
                title="Stop generating">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
              </button>
            } @else {
              <!-- Send button when not streaming -->
              <button
                class="send-btn"
                (click)="send()"
                [disabled]="!inputText.trim()"
                [class.active]="inputText.trim()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
            }
          </div>

          <p class="disclaimer">Erosolar uses Chain-of-Thought reasoning. Verify important info.</p>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .chat-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: var(--color-bg);
    }

    /* Messages area */
    .messages-area {
      flex: 1;
      overflow-y: auto;
      overflow-x: hidden;
    }

    .messages-inner {
      max-width: 48rem;
      margin: 0 auto;
      padding: 0 1rem;
    }

    /* Empty state */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: calc(100vh - 160px);
      padding: 2rem;

      h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--color-text);
        text-align: center;
        margin-bottom: 0.25rem;
      }

      .subtitle {
        font-size: 1.1rem;
        color: var(--color-accent);
        text-align: center;
        font-weight: 500;
        margin-bottom: 0.25rem;
      }

      .tagline {
        font-size: 0.9rem;
        color: var(--color-text-secondary);
        text-align: center;
        margin-bottom: 1rem;
      }

      .anti-openai {
        font-size: 0.85rem;
        color: var(--color-text-tertiary);
        text-align: center;
        font-style: italic;
        opacity: 0.8;
      }
    }

    /* Messages */
    .message {
      padding: 1.5rem 0;

      &.user {
        .message-row {
          justify-content: flex-end;
        }

        .message-content {
          max-width: 70%;
        }

        .user-message {
          background: var(--color-user-bg);
          border-radius: 1.5rem;
          padding: 0.75rem 1.25rem;
          white-space: pre-wrap;
          word-break: break-word;
        }
      }

      &.assistant {
        .message-row {
          align-items: flex-start;
          gap: 1rem;
        }

        .message-content {
          flex: 1;
          min-width: 0;
        }

        .prose {
          line-height: 1.75;
          color: var(--color-text);
        }
      }
    }

    .message-row {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
    }

    /* Avatars */
    .avatar {
      width: 28px;
      height: 28px;
      min-width: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .gpt-avatar {
      background: var(--color-accent);
      color: white;

      svg {
        width: 20px;
        height: 20px;
      }
    }

    .user-avatar {
      background: var(--color-surface-tertiary);
      color: var(--color-text);
    }

    /* Message actions */
    .message-actions {
      display: flex;
      gap: 0.25rem;
      margin-top: 0.75rem;
      opacity: 0;
      transition: opacity 0.15s ease;
    }

    .message:hover .message-actions {
      opacity: 1;
    }

    .action-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      border-radius: 6px;
      color: var(--color-text-tertiary);
      transition: background 0.15s ease, color 0.15s ease;

      &:hover {
        background: var(--color-surface-secondary);
        color: var(--color-text-secondary);
      }
    }

    /* Typing indicator */
    .typing-indicator {
      display: flex;
      gap: 4px;
      padding: 8px 0;

      span {
        width: 8px;
        height: 8px;
        background: var(--color-text-tertiary);
        border-radius: 50%;
        animation: typing 1.4s ease-in-out infinite;

        &:nth-child(1) { animation-delay: 0s; }
        &:nth-child(2) { animation-delay: 0.2s; }
        &:nth-child(3) { animation-delay: 0.4s; }
      }
    }

    @keyframes typing {
      0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.4;
      }
      30% {
        transform: translateY(-4px);
        opacity: 1;
      }
    }

    /* Input area */
    .input-area {
      padding: 0 1rem 1rem;
      background: var(--color-bg);
    }

    .input-wrapper {
      max-width: 48rem;
      margin: 0 auto;
    }

    .input-box {
      display: flex;
      align-items: flex-end;
      gap: 0.5rem;
      background: var(--color-surface-secondary);
      border: 1px solid var(--color-border);
      border-radius: 1.5rem;
      padding: 0.5rem 0.75rem;
      transition: border-color 0.15s ease;

      &:focus-within {
        border-color: var(--color-border-hover);
      }
    }

    textarea {
      flex: 1;
      background: none;
      border: none;
      color: var(--color-text);
      font-family: inherit;
      font-size: 1rem;
      line-height: 1.5;
      padding: 0.5rem 0.25rem;
      resize: none;
      min-height: 24px;
      max-height: 200px;

      &:focus {
        outline: none;
      }

      &::placeholder {
        color: var(--color-text-placeholder);
      }

      &:disabled {
        opacity: 0.6;
      }
    }

    .send-btn,
    .stop-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      min-width: 32px;
      border-radius: 50%;
      background: var(--color-text-tertiary);
      color: var(--color-bg);
      transition: background 0.15s ease, opacity 0.15s ease;

      &.active {
        background: var(--color-text);
      }

      &:disabled {
        opacity: 0.3;
        cursor: not-allowed;
      }
    }

    .stop-btn {
      background: var(--color-text);
      cursor: pointer;

      &:hover {
        opacity: 0.8;
      }
    }

    .disclaimer {
      text-align: center;
      font-size: 0.75rem;
      color: var(--color-text-tertiary);
      margin-top: 0.5rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .empty-state h1 {
        font-size: 1.5rem;
      }

      .message.user .message-content {
        max-width: 85%;
      }
    }
  `]
})
export class ChatComponent implements AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  @ViewChild('inputField') private inputField!: ElementRef;

  inputText = '';
  private shouldScroll = false;

  constructor(
    public chatService: ChatService,
    private sanitizer: DomSanitizer
  ) {}

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  async send(): Promise<void> {
    const text = this.inputText.trim();
    if (!text || this.chatService.isStreaming()) return;

    this.inputText = '';
    this.shouldScroll = true;

    if (this.inputField) {
      this.inputField.nativeElement.style.height = 'auto';
    }

    await this.chatService.sendMessage(text, false);
  }

  stop(): void {
    this.chatService.stopGeneration();
  }

  handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  autoResize(event: Event): void {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  }

  formatContent(content: string): SafeHtml {
    if (!content) return '';

    // Strip special tokens first
    let cleaned = content
      .replace(/<\|think_start\|>/g, '')
      .replace(/<\|think_end\|>/g, '')
      .replace(/<\|step\|>/g, '')
      .replace(/<\|answer\|>/g, '')
      .replace(/<\|end_turn\|>/g, '')
      .replace(/<\|im_start\|>/g, '')
      .replace(/<\|im_end\|>/g, '')
      .replace(/<\|pad\|>/g, '')
      .replace(/<\|endoftext\|>/g, '')
      .trim();

    // Parse markdown with syntax highlighting
    const html = markedInstance.parse(cleaned) as string;

    // Sanitize HTML to prevent XSS
    const sanitized = DOMPurify.sanitize(html, {
      ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li',
                     'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'a', 'span',
                     'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr'],
      ALLOWED_ATTR: ['href', 'target', 'rel', 'class']
    });

    return this.sanitizer.bypassSecurityTrustHtml(sanitized);
  }

  copyCode(code: string): void {
    navigator.clipboard.writeText(code).then(() => {
      // Could show a toast notification here
    });
  }

  copyMessage(content: string): void {
    // Strip tokens and copy plain text
    const cleaned = content
      .replace(/<\|[^|]+\|>/g, '')
      .trim();
    navigator.clipboard.writeText(cleaned);
  }

  private scrollToBottom(): void {
    try {
      const el = this.messagesContainer.nativeElement;
      el.scrollTop = el.scrollHeight;
    } catch (err) {}
  }
}
