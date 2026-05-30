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

/** The 5 openers (exact, from honest_pipeline.py). */
export const APPRECIATION_OPENERS: readonly string[] = [
  'thank you for', 'i am grateful for', 'i really appreciate',
  'i want to thank you for', 'we are thankful for',
];

/** The 16 impact clauses (exact, from honest_pipeline.py). */
export const APPRECIATION_IMPACTS: readonly string[] = [
  'made the work better', 'helped the whole team', 'raised the bar',
  'made things clearer', 'set a good example', 'moved the project forward',
  'made everyone feel welcome', 'saved us a lot of time',
  'made the result stronger', 'kept us on track', 'made the hard part easier',
  'lifted the whole mood', 'made the deadline reachable', 'inspired the rest of us',
  'made the review smoother', 'turned a hard week around',
];

/** The 6 optional closers (exact, from honest_pipeline.py). */
export const APPRECIATION_CLOSERS: readonly string[] = [
  'it did not go unnoticed', 'thank you again', 'it meant a lot to us',
  'please keep it up', 'the team noticed', 'it made a real difference',
];

export interface Appreciation {
  /** The chosen quality (one of APPRECIATION_QUALITIES). */
  quality: string;
  /** Raw, model-style text: lowercase, spaced punctuation (as the model emits it). */
  raw: string;
  /** Prettified, display-ready text (capitalized, normal spacing). */
  display: string;
}

function pick<T>(arr: readonly T[], rng: () => number): T {
  return arr[Math.floor(rng() * arr.length)];
}

/**
 * Prettify raw model-style text for display: collapse spaced punctuation,
 * normalize whitespace, and capitalize the first letter of each sentence.
 */
export function prettify(raw: string): string {
  let s = raw
    .replace(/\s+\./g, '.')   // " ." -> "."
    .replace(/\s+,/g, ',')    // " ," -> ","
    .replace(/\s+/g, ' ')     // collapse runs of whitespace
    .trim();
  // Capitalize the first letter, and the first letter after each sentence end.
  s = s.replace(/(^|[.!?]\s+)([a-z])/g, (_m, p1, p2) => p1 + p2.toUpperCase());
  // Capitalize the standalone pronoun "i".
  s = s.replace(/\bi\b/g, 'I');
  return s;
}

/**
 * Compose a single appreciation for `quality`, faithfully matching the model's
 * template:  "<opener> your <quality> . it <impact> . [<closer> .]".
 * `rng` defaults to Math.random (sampled for variety); pass a seeded RNG for
 * deterministic output.
 */
export function generateAppreciation(
  quality: string,
  rng: () => number = Math.random,
): Appreciation {
  const opener = pick(APPRECIATION_OPENERS, rng);
  const impact = pick(APPRECIATION_IMPACTS, rng);
  let body = `${opener} your ${quality} . it ${impact} .`;
  if (rng() < 0.5) {
    body += ` ${pick(APPRECIATION_CLOSERS, rng)} .`;
  }
  return { quality, raw: body, display: prettify(body) };
}

/** Pick a random quality (used by "Surprise me"). */
export function randomQuality(rng: () => number = Math.random): string {
  return pick(APPRECIATION_QUALITIES, rng);
}

/**
 * A small deterministic PRNG (mulberry32). Useful for reproducible generation;
 * matches the spirit of the Python pipeline's seeded random.Random.
 */
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
