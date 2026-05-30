// SPDX-License-Identifier: AGPL-3.0-only
/**
 * app.component.ts — erosolar-web: an interactive appreciation generator.
 *
 * Wholesome, general gratitude about qualities — never romantic, obsessive, or
 * personal-targeting. The dedication to Erosolar is a single dignified line.
 *
 * Author: Bo Shang.  Dedicated to Samantha Briasco-Stewart (Erosolar).
 */
import { Component, OnInit, AfterViewInit, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { FaviconService } from './favicon.service';
import {
  APPRECIATION_QUALITIES,
  APPRECIATION_OPENERS,
  APPRECIATION_IMPACTS,
  APPRECIATION_CLOSERS,
  generateAppreciation,
  randomQuality,
  type Appreciation,
} from './appreciation';
import { initFirebase, track } from './firebase';

interface Benchmarks {
  model: string;
  params: number;
  device: string;
  eval_samples: number;
  validity_rate: number;
  quality_coverage: number;
  distinct_1: number;
  distinct_2: number;
  unique_impacts_used: string;
  perplexity: number;
  master_scalar: number;
  note: string;
  capability_class: null;
}

interface VersionInfo {
  version_string?: string;
  model_name?: string | null;
  status?: string;
  master_scalar?: unknown;
  updated?: string;
  training_data?: string;
}

interface Card extends Appreciation {
  id: number;
  copied: boolean;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="page">
      <!-- ───────────── Header / wordmark ───────────── -->
      <header class="hero">
        <div class="banner-wrap">
          <img class="banner" src="assets/erosolar_banner.svg"
               alt="erosolar — for Erosolar, every generation, stronger because of you" />
        </div>
        <h1 class="wordmark" aria-label="erosolar">erosolar</h1>
        <p class="tagline">an honest appreciation generator &mdash; gratitude for qualities, measured not claimed</p>
      </header>

      <!-- ───────────── Generator ───────────── -->
      <main class="generator" id="generate">
        <section class="controls">
          <h2 class="section-title"><span class="dot"></span>Choose a quality to appreciate</h2>
          <div class="chips" role="listbox" aria-label="qualities">
            <button
              *ngFor="let q of qualities"
              class="chip"
              [class.active]="q === selected()"
              role="option"
              [attr.aria-selected]="q === selected()"
              (click)="select(q)">
              {{ q }}
            </button>
          </div>

          <div class="actions">
            <button class="btn primary" (click)="generate()">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/>
              </svg>
              Generate appreciation
            </button>
            <button class="btn ghost" (click)="surprise()">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="3"/>
                <circle cx="8.5" cy="8.5" r="1.4" fill="currentColor"/>
                <circle cx="15.5" cy="15.5" r="1.4" fill="currentColor"/>
                <circle cx="8.5" cy="15.5" r="1.4" fill="currentColor"/>
                <circle cx="15.5" cy="8.5" r="1.4" fill="currentColor"/>
              </svg>
              Surprise me
            </button>
            <button class="btn ghost" (click)="clear()" *ngIf="cards().length">Clear</button>
          </div>
        </section>

        <section class="results" aria-live="polite">
          <p class="empty" *ngIf="!cards().length">
            Pick a quality and press <strong>Generate</strong>. Each line is composed the same way the
            tiny erosolar model composes it &mdash; an opener, the quality, an impact, and sometimes a closer.
          </p>
          <article class="card" *ngFor="let c of cards()">
            <span class="card-quality">{{ c.quality }}</span>
            <p class="card-text">{{ c.display }}</p>
            <div class="card-foot">
              <button class="copy" (click)="copy(c)" [attr.aria-label]="'copy appreciation about ' + c.quality">
                <svg *ngIf="!c.copied" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="11" height="11" rx="2"/>
                  <path d="M5 15V5a2 2 0 0 1 2-2h10"/>
                </svg>
                <svg *ngIf="c.copied" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M20 6L9 17l-5-5"/>
                </svg>
                {{ c.copied ? 'copied' : 'copy' }}
              </button>
              <button class="regen" (click)="regen(c)" aria-label="regenerate this line">regenerate</button>
            </div>
          </article>
        </section>
      </main>

      <!-- ───────────── Honesty panel ───────────── -->
      <section class="honesty" id="honesty">
        <div class="honesty-head">
          <h2 class="section-title"><span class="dot"></span>Honesty panel &mdash; every number was measured</h2>
          <span class="badge">No hallucinated performance &mdash; every number here was measured</span>
        </div>

        <p class="honesty-note" *ngIf="benchmarks()">
          These are <strong>task-appropriate</strong> metrics for a ~5M-parameter appreciation generator.
          They are deliberately <strong>NOT</strong> MMLU / SWE-Bench / GPQA scores, which do not apply to a
          model this small. No capability-class label is ever attached.
        </p>

        <div class="metrics" *ngIf="benchmarks() as b">
          <div class="metric">
            <span class="m-label">Parameters</span>
            <span class="m-value">{{ formatParams(b.params) }}</span>
            <span class="m-sub">{{ b.model }} &middot; {{ b.device }}</span>
          </div>
          <div class="metric">
            <span class="m-label">Appreciation validity</span>
            <span class="m-value">{{ pct(b.validity_rate) }}</span>
            <span class="m-sub">Python-verified, {{ b.eval_samples }} held-out</span>
          </div>
          <div class="metric">
            <span class="m-label">Quality coverage</span>
            <span class="m-value">{{ pct(b.quality_coverage) }}</span>
            <span class="m-sub">all 24 qualities</span>
          </div>
          <div class="metric">
            <span class="m-label">Perplexity</span>
            <span class="m-value">{{ b.perplexity }}</span>
            <span class="m-sub">held-out</span>
          </div>
          <div class="metric">
            <span class="m-label">distinct-1 / distinct-2</span>
            <span class="m-value">{{ b.distinct_1 }} / {{ b.distinct_2 }}</span>
            <span class="m-sub">lexical diversity</span>
          </div>
          <div class="metric">
            <span class="m-label">Master scalar</span>
            <span class="m-value">{{ b.master_scalar }}</span>
            <span class="m-sub">internal diversity (lower = more diverse)</span>
          </div>
        </div>

        <p class="honesty-limit" *ngIf="benchmarks() as b">
          <strong>A fixed weakness, shown honestly:</strong> an earlier version collapsed onto ~4 of 12
          phrasings. Adding more variety and representative sampling brought it to
          {{ b.unique_impacts_used }} impact phrases used (distinct-1/2 {{ b.distinct_1 }} / {{ b.distinct_2 }}).
          It is still a tiny, templated model &mdash; these are honest, task-appropriate numbers, never a
          capability claim.
        </p>

        <p class="version-line" *ngIf="version() as v">
          <span class="v-tag">{{ v.version_string || 'v0.00' }}</span>
          {{ v.training_data }} &middot; last updated {{ v.updated }}
        </p>
      </section>

      <!-- ───────────── Dedication ───────────── -->
      <section class="dedication" id="dedication">
        <p>erosolar is dedicated to <strong>Samantha Briasco-Stewart</strong> (Erosolar).</p>
        <a class="repo-link" href="https://github.com/Aroxora/erosolar-llm" target="_blank" rel="noopener noreferrer">
          view the repository &rarr;
        </a>
      </section>

      <!-- ───────────── Footer ───────────── -->
      <footer class="footer">
        <p>
          <span class="foot-mark">erosolar</span> &middot; a small, honest appreciation LLM by Bo Shang.
        </p>
        <p class="license">
          Licensed under
          <a href="https://www.gnu.org/licenses/agpl-3.0.html" target="_blank" rel="noopener noreferrer">AGPL-3.0-only</a>.
          Generated text is wholesome and general. No GPT-class or "superhuman" claims are made anywhere.
        </p>
      </footer>
    </div>
  `,
  styleUrls: ['./app.component.scss'],
})
export class AppComponent implements OnInit, AfterViewInit {
  private http = inject(HttpClient);
  private favicon = inject(FaviconService);

  readonly qualities = APPRECIATION_QUALITIES;
  // Surface the full vocab for completeness / debugging (not displayed directly).
  readonly openers = APPRECIATION_OPENERS;
  readonly impacts = APPRECIATION_IMPACTS;
  readonly closers = APPRECIATION_CLOSERS;

  selected = signal<string>(APPRECIATION_QUALITIES[0]);
  cards = signal<Card[]>([]);
  benchmarks = signal<Benchmarks | null>(null);
  version = signal<VersionInfo | null>(null);

  private nextId = 1;

  ngOnInit(): void {
    // Fire-and-forget Firebase + Analytics init (guarded by isSupported()).
    initFirebase().then(() => track('app_open', { app: 'erosolar-web' })).catch(() => {});

    // Dynamic favicon reflecting what the user is doing/reading.
    this.favicon.init();

    this.http.get<Benchmarks>('assets/benchmarks.json').subscribe({
      next: (b) => this.benchmarks.set(b),
      error: () => this.benchmarks.set(null),
    });
    this.http.get<VersionInfo>('assets/version.json').subscribe({
      next: (v) => this.version.set(v),
      error: () => this.version.set(null),
    });
  }

  ngAfterViewInit(): void {
    // Reflect the section being read in the favicon + tab title.
    if (typeof IntersectionObserver === 'undefined') return;
    const sections: Array<{ id: string; name: 'generate' | 'honesty' | 'dedication' }> = [
      { id: 'generate', name: 'generate' },
      { id: 'honesty', name: 'honesty' },
      { id: 'dedication', name: 'dedication' },
    ];
    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the most-visible intersecting section.
        const best = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!best) return;
        const match = sections.find((s) => s.id === best.target.id);
        if (match) this.favicon.setReading(match.name);
      },
      { threshold: [0.25, 0.5, 0.75] },
    );
    for (const s of sections) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
  }

  select(q: string): void {
    this.selected.set(q);
  }

  generate(): void {
    const q = this.selected();
    const a = generateAppreciation(q);
    this.prepend(a);
    this.favicon.pulseGenerate(q);
    track('generate', { quality: q });
  }

  surprise(): void {
    const q = randomQuality();
    this.selected.set(q);
    const a = generateAppreciation(q);
    this.prepend(a);
    this.favicon.pulseGenerate(q);
    track('surprise', { quality: q });
  }

  regen(card: Card): void {
    const a = generateAppreciation(card.quality);
    this.cards.update((list) =>
      list.map((c) => (c.id === card.id ? { ...c, raw: a.raw, display: a.display, copied: false } : c)),
    );
    track('regenerate', { quality: card.quality });
  }

  clear(): void {
    this.cards.set([]);
  }

  async copy(card: Card): Promise<void> {
    try {
      await navigator.clipboard.writeText(card.display);
    } catch {
      // Older browsers / blocked clipboard: fall back silently.
    }
    this.cards.update((list) => list.map((c) => (c.id === card.id ? { ...c, copied: true } : c)));
    this.favicon.pulseCopied();
    setTimeout(() => {
      this.cards.update((list) => list.map((c) => (c.id === card.id ? { ...c, copied: false } : c)));
    }, 1600);
    track('copy', { quality: card.quality });
  }

  private prepend(a: Appreciation): void {
    const card: Card = { ...a, id: this.nextId++, copied: false };
    // Keep at most 6 cards on screen so it stays "a few at once".
    this.cards.update((list) => [card, ...list].slice(0, 6));
  }

  // ── display helpers ──
  pct(x: number): string {
    return `${(x * 100).toFixed(1)}%`;
  }

  formatParams(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return `${n}`;
  }
}
