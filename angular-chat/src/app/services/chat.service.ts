/**
 * Erosolar Chat Service
 * Handles conversation history, SSE streaming, and API communication
 *
 * Architecture:
 * - Firestore stores session data (public_sessions for anonymous, users/{uid}/sessions for auth)
 * - Cloud Run receives history with each request
 * - SSE streaming for real-time token-by-token generation
 *
 * Author: Bo Shang <bo@shang.software>
 */
import { Injectable, signal, computed, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { AnalyticsService } from './analytics.service';
import { SessionService } from './session.service';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  // For assistant messages, track thinking vs answer
  thinking?: string;
  answer?: string;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface ResponseAPIEvent {
  type: string;
  response?: {
    id: string;
    status: string;
    model: string;
  };
  item?: {
    id: string;
    type: string;
    status?: string;
  };
  delta?: string;
  item_id?: string;
  output_index?: number;
  content_index?: number;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  // Inject session service for Firestore persistence
  private sessionService = inject(SessionService);
  private analyticsService = inject(AnalyticsService);

  // Reactive state using Angular signals
  private messagesSignal = signal<ChatMessage[]>([]);
  private isStreamingSignal = signal<boolean>(false);

  // AbortController for cancelling requests
  private abortController: AbortController | null = null;

  // Public computed signals
  public messages = computed(() => this.messagesSignal());
  public isStreaming = computed(() => this.isStreamingSignal());
  public sessionId = computed(() => this.sessionService.currentSessionId());

  constructor(private http: HttpClient) {
    this.loadSession();
  }

  /**
   * Stop/cancel the current streaming response
   */
  stopGeneration(): void {
    void this.analyticsService.logEvent('chat_generation_stop');
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    // Mark the last message as not streaming
    this.messagesSignal.update(msgs => {
      const updated = [...msgs];
      const lastIdx = updated.length - 1;
      if (lastIdx >= 0 && updated[lastIdx].role === 'assistant' && updated[lastIdx].isStreaming) {
        updated[lastIdx] = {
          ...updated[lastIdx],
          isStreaming: false
        };
      }
      return updated;
    });
    this.isStreamingSignal.set(false);
    this.saveSession();
  }

  /**
   * Send a message and stream the response
   */
  async sendMessage(userContent: string, enableSearch: boolean = false): Promise<void> {
    if (this.isStreamingSignal()) return;

    const startTime = Date.now();
    let hadError = false;

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content: userContent,
      timestamp: new Date()
    };
    this.messagesSignal.update(msgs => [...msgs, userMessage]);
    void this.analyticsService.logEvent('chat_message_send', {
      message_length: userContent.length,
      has_search: enableSearch ? 1 : 0
    });

    // Create placeholder for assistant response
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      thinking: '',
      answer: '',
      timestamp: new Date(),
      isStreaming: true
    };
    this.messagesSignal.update(msgs => [...msgs, assistantMessage]);

    this.isStreamingSignal.set(true);

    try {
      // Build messages array for API (conversation history)
      const apiMessages = this.messagesSignal()
        .filter(m => !m.isStreaming)
        .map(m => ({
          role: m.role,
          content: m.role === 'assistant'
            ? (this.formatDisplayContent(m.thinking ?? '', m.answer ?? '') || m.content)
            : m.content
        }));

      await this.streamCompletion(apiMessages, enableSearch);
    } catch (error) {
      hadError = true;
      void this.analyticsService.logEvent('chat_request_error', {
        error_name: error instanceof Error ? error.name : 'unknown'
      });
      console.error('Chat error:', error);
      this.updateLastAssistantMessage(
        `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        false
      );
    } finally {
      void this.analyticsService.logEvent('chat_response_complete', {
        duration_ms: Date.now() - startTime,
        success: hadError ? 0 : 1
      });
      this.isStreamingSignal.set(false);
      this.saveSession();
    }
  }

  /**
   * Stream completion from Cloud Run API using Response API SSE
   */
  private async streamCompletion(
    messages: Array<{ role: string; content: string }>,
    enableSearch: boolean
  ): Promise<void> {
    const url = `${environment.apiUrl}${environment.chatCompletionsPath}`;

    // Create new AbortController for this request
    this.abortController = new AbortController();

    // Convert messages to Response API input format
    const input = messages.map(m => ({
      role: m.role,
      content: [{ type: 'input_text', text: m.content }]
    }));

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: environment.modelName,
        input,
        stream: true,
        max_output_tokens: 500,
        temperature: 0.7
      }),
      signal: this.abortController.signal
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let thinkingContent = '';
    let answerContent = '';
    let inThinking = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process SSE events (format: "event: type\ndata: {...}\n\n")
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const eventBlock of events) {
        if (!eventBlock.trim()) continue;

        const lines = eventBlock.split('\n');
        let eventType = '';
        let eventData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7);
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6);
          }
        }

        if (!eventData) continue;

        try {
          const event: ResponseAPIEvent = JSON.parse(eventData);

          if (event.type === 'response.output_item.added') {
            if (event.item?.type === 'reasoning') {
              inThinking = true;
            } else if (event.item?.type === 'message') {
              inThinking = false;
            }
          } else if (event.type === 'response.output_text.delta') {
            const delta = event.delta || '';
            if (inThinking) {
              thinkingContent += delta;
            } else {
              answerContent += delta;
            }
            // Update message with current content
            const fullContent = this.formatDisplayContent(thinkingContent, answerContent);
            this.updateLastAssistantMessage(fullContent, true, thinkingContent, answerContent);
          } else if (event.type === 'response.output_item.done') {
            if (event.item?.type === 'reasoning') {
              inThinking = false;
            }
          } else if (event.type === 'response.completed') {
            break;
          }
        } catch (e) {
          // Ignore parse errors for partial data
        }
      }
    }

    // Finalize message
    const fullContent = this.formatDisplayContent(thinkingContent, answerContent);
    this.updateLastAssistantMessage(fullContent, false, thinkingContent, answerContent);
  }

  /**
   * Format content for display with thinking tokens
   */
  private formatDisplayContent(thinking: string, answer: string): string {
    const parts: string[] = [];
    if (thinking) {
      parts.push(`<|think_start|>\n${thinking}\n<|think_end|>`);
    }
    if (answer) {
      parts.push(`<|answer|>\n${answer}`);
    }
    return parts.join('\n');
  }

  /**
   * Parse CoT tokens to separate thinking from answer
   * Tokens: <|think_start|>, <|step|>, <|answer|>, <|end_turn|>
   */
  private parseThinkingAndAnswer(content: string): { thinking: string; answer: string } {
    let thinking = '';
    let answer = '';

    // Extract thinking section
    const thinkStartMatch = content.match(/<\|think_start\|>([\s\S]*?)(?:<\|answer\|>|$)/);
    if (thinkStartMatch) {
      thinking = thinkStartMatch[1]
        .replace(/<\|step\|>/g, '\n\n') // Convert step tokens to line breaks
        .trim();
    }

    // Extract answer section
    const answerMatch = content.match(/<\|answer\|>([\s\S]*?)(?:<\|end_turn\|>|$)/);
    if (answerMatch) {
      answer = answerMatch[1].trim();
    } else if (!content.includes('<|think_start|>')) {
      // No CoT tokens, treat entire content as answer
      answer = content;
    }

    return { thinking, answer };
  }

  /**
   * Update the last assistant message
   */
  private updateLastAssistantMessage(
    content: string,
    isStreaming: boolean,
    thinking?: string,
    answer?: string
  ): void {
    this.messagesSignal.update(msgs => {
      const updated = [...msgs];
      const lastIdx = updated.length - 1;
      if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
        updated[lastIdx] = {
          ...updated[lastIdx],
          content,
          thinking: thinking ?? updated[lastIdx].thinking,
          answer: answer ?? updated[lastIdx].answer,
          isStreaming
        };
      }
      return updated;
    });
  }

  /**
   * Search the web using Tavily
   */
  async searchWeb(query: string): Promise<any> {
    void this.analyticsService.logEvent('chat_search_request', {
      query_length: query.length
    });
    const url = `${environment.apiUrl}${environment.searchPath}`;
    return this.http.post(url, {
      query,
      max_results: 5
    }).toPromise();
  }

  /**
   * Start a new chat session
   */
  async newSession(): Promise<void> {
    // Save current session first if it has messages
    if (this.messagesSignal().length > 0) {
      await this.saveSession();
    }
    // Clear messages and create new session
    this.messagesSignal.set([]);
    await this.sessionService.newSession();
    // Refresh sessions list
    await this.sessionService.loadUserSessions();
    void this.analyticsService.logEvent('chat_session_new');
  }

  /**
   * Save current session to Firestore
   */
  private async saveSession(): Promise<void> {
    const messages = this.messagesSignal().map(m => ({
      role: m.role,
      content: m.content,
      thinking: m.thinking,
      answer: m.answer,
      timestamp: m.timestamp
    }));

    try {
      await this.sessionService.saveMessages(messages);
    } catch (e) {
      console.error('Failed to save session:', e);
    }
  }

  /**
   * Load session from Firestore
   */
  private async loadSession(): Promise<void> {
    try {
      const sessionId = this.sessionService.getSessionId();
      if (sessionId) {
        const session = await this.sessionService.loadSession(sessionId);
        if (session && session.messages) {
          this.messagesSignal.set(session.messages.map(m => ({
            role: m.role as 'user' | 'assistant' | 'system',
            content: m.content,
            thinking: m.thinking,
            answer: m.answer,
            timestamp: m.timestamp,
            isStreaming: false
          })));
        }
      }
    } catch (e) {
      console.error('Failed to load session:', e);
    }
  }

  /**
   * Get all saved sessions (from session service)
   */
  getAllSessions(): ChatSession[] {
    return this.sessionService.sessions();
  }

  /**
   * Load a specific session by ID
   */
  async loadSessionById(id: string): Promise<void> {
    await this.sessionService.switchSession(id);
    await this.loadSession();
  }

  /**
   * Delete a session
   */
  async deleteSession(id: string): Promise<void> {
    await this.sessionService.deleteSession(id);
    if (id === this.sessionService.getSessionId()) {
      this.messagesSignal.set([]);
    }
    void this.analyticsService.logEvent('chat_session_delete');
  }

  /**
   * Clear conversation history
   */
  async clearHistory(): Promise<void> {
    this.messagesSignal.set([]);
    await this.saveSession();
  }
}
