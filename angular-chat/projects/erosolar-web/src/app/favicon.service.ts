// SPDX-License-Identifier: AGPL-3.0-only
/**
 * favicon.service.ts — a dynamic, canvas-rendered favicon that reflects what the
 * user is doing or reading, in the erosolar solar palette.
 *
 * States:
 *   - idle / reading: a gently breathing sun. The section being read tints it and
 *     adds a small corner glyph (generator = plain sun, honesty = green ✓ badge,
 *     dedication = a warm full-bright sun).
 *   - generating: the corona spins fast and the chosen quality's initial appears
 *     in the disc; the tab title shows "erosolar · <quality>".
 *   - copied: a brief green check.
 *
 * Everything is drawn to an offscreen canvas and pushed to <link rel="icon"> as a
 * PNG data URL, throttled to ~12fps. No images, no network — pure canvas.
 */
import { Injectable } from '@angular/core';

type Section = 'generate' | 'honesty' | 'dedication';
type Activity = { kind: 'idle' } | { kind: 'generating'; letter: string } | { kind: 'copied' };

const GOLD = '#ffd27a';
const AMBER = '#ffab2e';
const EMBER = '#e9711c';
const VERIFY = '#5fd38a';

@Injectable({ providedIn: 'root' })
export class FaviconService {
  private canvas?: HTMLCanvasElement;
  private ctx?: CanvasRenderingContext2D | null;
  private link?: HTMLLinkElement;

  private size = 64;
  private section: Section = 'generate';
  private activity: Activity = { kind: 'idle' };
  private activityUntil = 0;
  private lastPush = 0;
  private running = false;

  private readonly baseTitle = 'erosolar — honest appreciation generator';

  /** Create the canvas + <link rel="icon"> and start the animation loop. Safe to call once. */
  init(): void {
    if (this.running || typeof document === 'undefined') return;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = this.size;
    this.ctx = this.canvas.getContext('2d');

    let link = document.querySelector<HTMLLinkElement>('link[rel~="icon"]');
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    link.type = 'image/png';
    this.link = link;

    this.running = true;
    const loop = () => {
      if (!this.running) return;
      this.draw(performance.now());
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  /** The user generated appreciation about `quality`: spin up + show its initial. */
  pulseGenerate(quality: string): void {
    this.activity = { kind: 'generating', letter: (quality[0] || 'e').toUpperCase() };
    this.activityUntil = performance.now() + 1900;
    this.setTitle(`erosolar · ${quality}`);
  }

  /** The user copied a line: flash a green check. */
  pulseCopied(): void {
    this.activity = { kind: 'copied' };
    this.activityUntil = performance.now() + 1400;
    this.setTitle('erosolar · copied ✓');
  }

  /** The user is reading a section: tints the idle icon + title. */
  setReading(section: Section): void {
    this.section = section;
    if (this.activity.kind === 'idle') {
      const label = section === 'honesty' ? 'honesty panel'
        : section === 'dedication' ? 'a dedication' : 'appreciation';
      this.setTitle(`erosolar · ${label}`);
    }
  }

  // ── rendering ──

  private draw(now: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    if (now > this.activityUntil && this.activity.kind !== 'idle') {
      this.activity = { kind: 'idle' };
      this.setTitle(this.baseTitle);
    }
    const s = this.size;
    const cx = s / 2, cy = s / 2;
    ctx.clearRect(0, 0, s, s);

    const generating = this.activity.kind === 'generating';
    const copied = this.activity.kind === 'copied';

    // Breathing / spin parameters.
    const pulse = 1 + 0.06 * Math.sin(now / 480);
    const spin = generating ? now / 130 : now / 2600;
    const rays = 12;
    const discR = (generating ? 15 : 14) * pulse;
    const rayInner = discR + 2;
    const rayOuter = (generating ? 28 : 24) * pulse;

    if (copied) {
      this.drawCheck(ctx, cx, cy, s);
      this.pushThrottled(now);
      return;
    }

    // Corona rays.
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(spin);
    ctx.lineCap = 'round';
    for (let i = 0; i < rays; i++) {
      const a = (i / rays) * Math.PI * 2;
      const flick = generating ? 0.7 + 0.3 * Math.sin(now / 90 + i) : 0.45 + 0.25 * Math.sin(now / 700 + i);
      ctx.globalAlpha = flick;
      ctx.strokeStyle = i % 2 === 0 ? AMBER : GOLD;
      ctx.lineWidth = generating ? 3.2 : 2.4;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * rayInner, Math.sin(a) * rayInner);
      ctx.lineTo(Math.cos(a) * rayOuter, Math.sin(a) * rayOuter);
      ctx.stroke();
    }
    ctx.restore();

    // Sun disc.
    ctx.globalAlpha = 1;
    const warm = this.section === 'dedication';
    const grad = ctx.createRadialGradient(cx - 3, cy - 4, 2, cx, cy, discR);
    grad.addColorStop(0, warm ? '#fff0cf' : GOLD);
    grad.addColorStop(0.6, AMBER);
    grad.addColorStop(1, EMBER);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, discR, 0, Math.PI * 2);
    ctx.fill();

    if (generating && this.activity.kind === 'generating') {
      // The quality's initial, dark over the gold disc.
      ctx.fillStyle = '#231708';
      ctx.font = `bold ${Math.round(discR * 1.5)}px "Courier New", monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(this.activity.letter, cx, cy + 1);
    } else if (this.section === 'honesty') {
      // Small "verified" check badge, bottom-right.
      this.drawBadgeCheck(ctx, s - 15, s - 15, 11);
    }

    this.pushThrottled(now);
  }

  private drawCheck(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number): void {
    ctx.beginPath();
    ctx.arc(cx, cy, s * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = VERIFY;
    ctx.fill();
    ctx.strokeStyle = '#0d2b18';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(cx - 11, cy + 1);
    ctx.lineTo(cx - 3, cy + 9);
    ctx.lineTo(cx + 12, cy - 10);
    ctx.stroke();
  }

  private drawBadgeCheck(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number): void {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = VERIFY;
    ctx.fill();
    ctx.strokeStyle = '#0d2b18';
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(cx - 4.5, cy + 0.5);
    ctx.lineTo(cx - 1, cy + 4);
    ctx.lineTo(cx + 5, cy - 4);
    ctx.stroke();
  }

  private pushThrottled(now: number): void {
    if (!this.canvas || !this.link) return;
    if (now - this.lastPush < 80) return; // ~12fps
    this.lastPush = now;
    this.link.href = this.canvas.toDataURL('image/png');
  }

  private setTitle(t: string): void {
    if (typeof document !== 'undefined') document.title = t;
  }
}
