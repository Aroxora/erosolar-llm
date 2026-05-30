// SPDX-License-Identifier: AGPL-3.0-only
/**
 * appreciation.ts — in-browser, license-clean reproduction of the erosolar
 * appreciation generator.
 *
 * The real model is a tiny PyTorch transformer whose generation is template-based
 * (see honest_pipeline.py: appreciation_corpus / valid_appreciation). This file
 * reproduces that exact vocabulary and composition rule in TypeScript so the app
 * can generate appreciation entirely client-side, with no backend.
 *
 * It is deliberately about QUALITIES and craft — gratitude one could sincerely
 * offer any colleague — never about any one person, never romantic, never
 * obsessive, never personal-targeting.
 */

/** The 24 qualities the model was trained to appreciate (exact, from honest_pipeline.py). */
export const APPRECIATION_QUALITIES: readonly string[] = [
  'care', 'craft', 'effort', 'patience', 'clarity', 'curiosity',
  'kindness', 'rigor', 'generosity', 'courage', 'focus', 'honesty',
  'diligence', 'creativity', 'leadership', 'humility', 'persistence',
  'insight', 'warmth', 'integrity', 'dedication', 'attention',
  'teamwork', 'optimism',
];

/** The 8 openers (exact, from honest_pipeline.py). */
export const APPRECIATION_OPENERS: readonly string[] = [
  'thank you for', 'i am grateful for', 'i really appreciate',
  'i want to thank you for', 'we are thankful for', "i'm so thankful for",
  'i deeply appreciate', 'many thanks for',
];

/**
 * Impacts mapped to the qualities they actually FIT (exact, from honest_pipeline.py).
 * This keeps each appreciation coherent ("clarity ... made things clearer") instead of
 * mismatched, which is the quality-agnostic-impact weakness the LLM judge surfaced.
 */
export const QUALITY_IMPACTS: Readonly<Record<string, readonly string[]>> = {
  care:        ['helped the whole team', 'made everyone feel welcome', 'lifted the whole mood', "made someone's day"],
  craft:       ['made the work better', 'made the result stronger', 'raised the bar', 'made the review smoother'],
  effort:      ['moved the project forward', 'made the deadline reachable', 'made the hard part easier', 'kept us on track'],
  patience:    ['made the hard part easier', 'kept us on track', 'kept everyone calm', 'gave us room to get it right'],
  clarity:     ['made things clearer', 'made the review smoother', 'saved us a lot of time', 'made the work better'],
  curiosity:   ['moved the project forward', 'made the result stronger', 'inspired the rest of us', 'made things clearer'],
  kindness:    ['made everyone feel welcome', 'lifted the whole mood', 'helped the whole team', "made someone's day"],
  rigor:       ['made the work better', 'raised the bar', 'made the result stronger', 'caught the problems early'],
  generosity:  ['helped the whole team', 'made everyone feel welcome', 'lifted the whole mood', 'shared the load'],
  courage:     ['set a good example', 'inspired the rest of us', 'raised the bar', 'turned a hard week around'],
  focus:       ['moved the project forward', 'kept us on track', 'made the deadline reachable', 'saved us a lot of time'],
  honesty:     ['earned our trust', 'set a good example', 'kept us honest', 'made things clearer'],
  diligence:   ['moved the project forward', 'kept us on track', 'made the deadline reachable', 'made the work better'],
  creativity:  ['made the result stronger', 'inspired the rest of us', 'moved the project forward', 'raised the bar'],
  leadership:  ['set a good example', 'kept us on track', 'inspired the rest of us', 'moved the project forward'],
  humility:    ['made everyone feel welcome', 'set a good example', 'shared the load', 'helped the whole team'],
  persistence: ['made the deadline reachable', 'made the hard part easier', 'kept us on track', 'turned a hard week around'],
  insight:     ['made things clearer', 'made the result stronger', 'caught the problems early', 'saved us a lot of time'],
  warmth:      ['made everyone feel welcome', 'lifted the whole mood', 'helped the whole team', "made someone's day"],
  integrity:   ['set a good example', 'earned our trust', 'raised the bar', 'kept us honest'],
  dedication:  ['moved the project forward', 'made the deadline reachable', 'kept us on track', 'made the result stronger'],
  attention:   ['made the review smoother', 'made things clearer', 'caught the problems early', 'made the work better'],
  teamwork:    ['helped the whole team', 'made everyone feel welcome', 'kept us on track', 'moved the project forward'],
  optimism:    ['lifted the whole mood', 'turned a hard week around', 'inspired the rest of us', 'made everyone feel welcome'],
};

/** Global impact pool (union) — for reference/completeness. */
export const APPRECIATION_IMPACTS: readonly string[] = Array.from(
  new Set(Object.values(QUALITY_IMPACTS).flat()),
).sort();

/** The 8 optional closers (exact, from honest_pipeline.py). */
export const APPRECIATION_CLOSERS: readonly string[] = [
  'it did not go unnoticed', 'thank you again', 'it meant a lot to us',
  'please keep it up', 'the team noticed', 'it made a real difference',
  'we see it', 'that is rare',
];

export interface Appreciation {
  quality: string;
  raw: string;
  display: string;
}

function pick<T>(arr: readonly T[], rng: () => number): T {
  return arr[Math.floor(rng() * arr.length)];
}

/** Prettify raw model-style text: collapse spaced punctuation and capitalize sentences. */
export function prettify(raw: string): string {
  let s = raw
    .replace(/\s+\./g, '.')
    .replace(/\s+,/g, ',')
    .replace(/\s+/g, ' ')
    .trim();
  s = s.replace(/(^|[.!?]\s+)([a-z])/g, (_m, p1, p2) => p1 + p2.toUpperCase());
  s = s.replace(/\bi\b/g, 'I');
  return s;
}

/**
 * Compose an appreciation for `quality`, matching the model's two template forms:
 *   T1: "<opener> your <quality> . it <impact> . [<closer> .]"
 *   T2: "your <quality> <impact> . [<closer> .]"
 * The impact is drawn from the impacts that actually FIT the quality, so the line
 * is coherent. `rng` defaults to Math.random; pass a seeded RNG for determinism.
 */
export function generateAppreciation(
  quality: string,
  rng: () => number = Math.random,
): Appreciation {
  const impacts = QUALITY_IMPACTS[quality] ?? APPRECIATION_IMPACTS;
  const impact = pick(impacts, rng);
  let body: string;
  if (rng() < 0.5) {
    body = `${pick(APPRECIATION_OPENERS, rng)} your ${quality} . it ${impact} .`;
  } else {
    body = `your ${quality} ${impact} .`;
  }
  if (rng() < 0.5) {
    body += ` ${pick(APPRECIATION_CLOSERS, rng)} .`;
  }
  return { quality, raw: body, display: prettify(body) };
}

/** Pick a random quality (used by "Surprise me"). */
export function randomQuality(rng: () => number = Math.random): string {
  return pick(APPRECIATION_QUALITIES, rng);
}

/** Deterministic PRNG (mulberry32) for reproducible generation. */
export function seededRng(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
