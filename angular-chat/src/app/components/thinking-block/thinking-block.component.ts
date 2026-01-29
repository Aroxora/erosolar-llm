/**
 * Thinking Block Component
 * Displays reasoning in a collapsible block
 * Timer freezes when streaming stops
 *
 * Author: Bo Shang <bo@shang.software>
 */
import { Component, Input, signal, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-thinking-block',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="thinking-block" [class.collapsed]="isCollapsed()" [class.streaming]="isStreaming">
      <button class="thinking-header" (click)="toggle()">
        <span class="thinking-icon">
          @if (isStreaming) {
            <svg class="sparkle-icon" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z"/>
            </svg>
          } @else {
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
            </svg>
          }
        </span>
        <span class="thinking-label">
          @if (isStreaming) {
            Thinking...
          } @else {
            Thought for {{ finalDuration }}
          }
        </span>
        <span class="thinking-toggle">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </span>
      </button>

      <div class="thinking-content" [class.hidden]="isCollapsed()">
        <div class="thinking-text">{{ cleanThinking }}</div>
      </div>
    </div>
  `,
  styles: [`
    .thinking-block {
      margin: 12px 0;
      border-radius: 8px;
      background: var(--color-surface-secondary);
      overflow: hidden;
    }

    .thinking-header {
      display: flex;
      align-items: center;
      gap: 8px;
      width: 100%;
      padding: 12px 16px;
      background: none;
      border: none;
      color: var(--color-text-secondary);
      font-family: inherit;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.15s ease;

      &:hover {
        background: var(--color-surface-tertiary);
      }
    }

    .thinking-icon {
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--color-accent);

      .sparkle-icon {
        animation: sparkle 1.5s ease-in-out infinite;
      }
    }

    @keyframes sparkle {
      0%, 100% {
        transform: scale(1) rotate(0deg);
        opacity: 1;
      }
      50% {
        transform: scale(1.1) rotate(180deg);
        opacity: 0.8;
      }
    }

    .thinking-label {
      flex: 1;
      text-align: left;
      font-weight: 500;
    }

    .thinking-toggle {
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s ease;
    }

    .collapsed .thinking-toggle {
      transform: rotate(-90deg);
    }

    .thinking-content {
      padding: 0 16px 16px;
      max-height: 300px;
      overflow-y: auto;
      transition: all 0.2s ease;

      &.hidden {
        max-height: 0;
        padding-top: 0;
        padding-bottom: 0;
        overflow: hidden;
      }
    }

    .thinking-text {
      font-size: 13px;
      line-height: 1.6;
      color: var(--color-text-secondary);
      white-space: pre-wrap;
      word-break: break-word;
    }

    .thinking-content::-webkit-scrollbar {
      width: 4px;
    }

    .thinking-content::-webkit-scrollbar-thumb {
      background: var(--color-border);
      border-radius: 2px;
    }
  `]
})
export class ThinkingBlockComponent implements OnChanges {
  @Input() thinking: string = '';
  @Input() isStreaming: boolean = false;

  isCollapsed = signal(true);
  finalDuration: string = '0s';

  private startTime: number = Date.now();
  private wasStreaming: boolean = false;

  /**
   * Strip special tokens from thinking content
   */
  get cleanThinking(): string {
    if (!this.thinking) return '';
    return this.thinking
      .replace(/<\|think_start\|>/g, '')
      .replace(/<\|think_end\|>/g, '')
      .replace(/<\|step\|>/g, '\n')
      .replace(/<\|answer\|>/g, '')
      .replace(/<\|end_turn\|>/g, '')
      .replace(/<\|im_start\|>/g, '')
      .replace(/<\|im_end\|>/g, '')
      .replace(/<\|pad\|>/g, '')
      .replace(/<\|endoftext\|>/g, '')
      .trim();
  }

  ngOnChanges(changes: SimpleChanges): void {
    // When streaming starts, record start time
    if (changes['isStreaming']) {
      if (this.isStreaming && !this.wasStreaming) {
        // Just started streaming
        this.startTime = Date.now();
      } else if (!this.isStreaming && this.wasStreaming) {
        // Just stopped streaming - freeze the duration
        this.finalDuration = this.calculateDuration();
      }
      this.wasStreaming = this.isStreaming;
    }
  }

  private calculateDuration(): string {
    const elapsed = Date.now() - this.startTime;
    const seconds = Math.round(elapsed / 1000);
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  }

  toggle(): void {
    this.isCollapsed.update(v => !v);
  }
}
