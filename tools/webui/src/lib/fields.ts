// field descriptor table: single source of truth for AceRequest field knowledge.
//
// every field in AceRequest is listed here with its section and type.
// helpers derive clear, preserve, serialize, and pick logic from this table.
// adding a field = adding one line here.

import type { AceRequest, AceTaskInfo } from './types.js';

export type FieldSection =
	| 'content'
	| 'metadata'
	| 'lm'
	| 'flow'
	| 'advanced'
	| 'toolbar'
	| 'routing';

interface FieldDef {
	key: keyof AceRequest;
	section: FieldSection;
	type: 'str' | 'num' | 'bool';
}

export const FIELDS: readonly FieldDef[] = [
	// content: per-variation, replaced by LM results
	{ key: 'caption', section: 'content', type: 'str' },
	{ key: 'lyrics', section: 'content', type: 'str' },
	{ key: 'audio_codes', section: 'content', type: 'str' },
	{ key: 'use_cot_caption', section: 'content', type: 'bool' },

	// metadata: per-variation, cleared by Clear metadata
	{ key: 'vocal_language', section: 'metadata', type: 'str' },
	{ key: 'bpm', section: 'metadata', type: 'num' },
	{ key: 'duration', section: 'metadata', type: 'num' },
	{ key: 'keyscale', section: 'metadata', type: 'str' },
	{ key: 'timesignature', section: 'metadata', type: 'str' },

	// LM control: user settings, preserved across Compose/variation switch
	{ key: 'lm_batch_size', section: 'lm', type: 'num' },
	{ key: 'lm_temperature', section: 'lm', type: 'num' },
	{ key: 'lm_cfg_scale', section: 'lm', type: 'num' },
	{ key: 'lm_top_p', section: 'lm', type: 'num' },
	{ key: 'lm_top_k', section: 'lm', type: 'num' },
	{ key: 'lm_negative_prompt', section: 'lm', type: 'str' },
	{ key: 'lm_seed', section: 'metadata', type: 'num' },

	// flow matching: cleared by Clear flow matching, preserved across Compose
	{ key: 'inference_steps', section: 'flow', type: 'num' },
	{ key: 'guidance_scale', section: 'flow', type: 'num' },
	{ key: 'shift', section: 'flow', type: 'num' },
	{ key: 'audio_cover_strength', section: 'flow', type: 'num' },
	{ key: 'cover_noise_strength', section: 'flow', type: 'num' },
	{ key: 'repainting_start', section: 'flow', type: 'num' },
	{ key: 'repainting_end', section: 'flow', type: 'num' },
	{ key: 'seed', section: 'flow', type: 'num' },

	// advanced and post-processing: cleared by its own Clear, preserved across Compose
	{ key: 'custom_timesteps', section: 'advanced', type: 'str' },
	{ key: 'dcw_mode', section: 'advanced', type: 'str' },
	{ key: 'dcw_scaler', section: 'advanced', type: 'num' },
	{ key: 'dcw_high_scaler', section: 'advanced', type: 'num' },
	{ key: 'solver', section: 'advanced', type: 'str' },
	{ key: 'stork_substeps', section: 'advanced', type: 'num' },
	{ key: 'latent_shift', section: 'advanced', type: 'num' },
	{ key: 'latent_rescale', section: 'advanced', type: 'num' },
	{ key: 'peak_clip', section: 'advanced', type: 'num' },
	{ key: 'mp3_bitrate', section: 'advanced', type: 'num' },

	// toolbar: preserved, not in any clear section
	{ key: 'synth_batch_size', section: 'toolbar', type: 'num' },

	// routing: model and task selection, preserved across Compose
	{ key: 'task_type', section: 'routing', type: 'str' },
	{ key: 'track', section: 'routing', type: 'str' },
	{ key: 'synth_model', section: 'routing', type: 'str' },
	{ key: 'lm_model', section: 'routing', type: 'str' },
	{ key: 'adapter', section: 'routing', type: 'str' },
	{ key: 'adapter_scale', section: 'routing', type: 'num' },
	{ key: 'vae', section: 'routing', type: 'str' }
];

// convert to number, undefined if empty/NaN
export function num(v: unknown): number | undefined {
	if (v == null || String(v).trim() === '') return undefined;
	const n = Number(v);
	return isNaN(n) ? undefined : n;
}

// typed dynamic access helpers
type Rec = Record<string, unknown>;
function get(r: AceRequest | Partial<AceRequest>, key: keyof AceRequest): unknown {
	return (r as Rec)[key];
}
function set(r: AceRequest | Partial<AceRequest>, key: keyof AceRequest, val: unknown): void {
	(r as Rec)[key] = val;
}

// resolve a field value to its serialized form, undefined if empty
function resolveField(f: FieldDef, raw: unknown): unknown {
	if (f.type === 'num') {
		const n = num(raw);
		return n != null ? n : undefined;
	}
	if (f.type === 'str') return raw ? String(raw) : undefined;
	return raw ?? undefined;
}

// serialize non-empty fields for JSON export.
// caller injects srcRange (repainting_start/end) and validates adapter.
export function buildSparse(r: AceRequest): AceRequest {
	const out: AceRequest = { caption: String(r.caption || '') };
	for (const f of FIELDS) {
		if (f.key === 'caption') continue;
		const val = resolveField(f, get(r, f.key));
		if (val !== undefined) set(out, f.key, val);
	}
	return out;
}

// reset all fields in a section to empty/undefined.
// caller handles srcRange for 'flow'.
export function clearSection(r: AceRequest, section: FieldSection): void {
	for (const f of FIELDS) {
		if (f.section !== section) continue;
		set(r, f.key, f.type === 'str' ? '' : undefined);
	}
}

// merge content from incoming with settings from current.
// content and metadata come from incoming, everything else from current.
export function withCurrentSettings(incoming: AceRequest, current: AceRequest): AceRequest {
	const out = { ...incoming };
	for (const f of FIELDS) {
		if (f.section === 'content' || f.section === 'metadata') continue;
		set(out, f.key, get(current, f.key));
	}
	return out;
}

// the sections a /synth batch shares: the form's, for every request.
export const SYNTH_SECTIONS: readonly FieldSection[] = ['flow', 'advanced', 'toolbar', 'routing'];

// one request of a batch: r's own fields, with every field of the shared
// sections taken from shared -- a pickSections() result -- and none kept
// from r. A field cleared in the form is gone from every request, not left
// as the copy the LM echoed into r (cadenza-audio #1164).
function batchRequest(r: AceRequest, shared: Partial<AceRequest>): AceRequest {
	const out = { ...r } as Rec;
	const drop = new Set(SYNTH_SECTIONS);
	for (const f of FIELDS) {
		if (drop.has(f.section)) delete out[f.key];
	}
	return { ...out, ...shared } as AceRequest;
}

// a request's audio codes as the webui sends them: the white space a paste
// leaves trimmed, and any inside -- a line break, a tab -- made a space,
// which the server skips between codes where it stops at a break
// (parse_csv). So a stray newline is no codes, as the form's "LM codes"
// light shows it, and codes wrapped over lines are sent whole
// (cadenza-audio #1182).
export function codesOf(r: Partial<AceRequest>): string {
	return String(r.audio_codes ?? '')
		.replace(/\s+/g, ' ')
		.trim();
}

// a request as an export writes it: its codes as a synth would send them
// (codesOf), so the file says what would run; none for codes that are none.
// The form and the pending requests keep the codes as typed.
export function asSent(r: AceRequest): AceRequest {
	const out = { ...r };
	const codes = codesOf(r);
	if (codes) out.audio_codes = codes;
	else delete out.audio_codes;
	return out;
}

// the range of the int the server reads a code into (ops_codes_given's
// from_chars)
const CODE_MIN = -(2 ** 31);
const CODE_MAX = 2 ** 31 - 1;

// the server's rule for codes that are given: whether one is held -- past
// leading commas and spaces, an integer starts that fits an int, as
// ops_codes_given reads it with from_chars, so a run of digits past an
// int's range holds none (cadenza-audio #1182)
export function holdsCode(codes: string): boolean {
	const first = /^-?[0-9]+/.exec(codes.replace(/^[, ]*/, ''));
	if (!first) return false;
	const v = Number(first[0]);
	return v >= CODE_MIN && v <= CODE_MAX;
}

// the largest seed a take runs: the engine's seeds are [0, UINT32_MAX], and a
// larger one runs as its low 32 bits (request.h), so another take's twin
export const SEED_MAX = 2 ** 32 - 1;

// a number as the server reads one into an integer: whole, toward zero
function whole(n: number | undefined): number | undefined {
	return n == null ? undefined : Math.trunc(n);
}

// the most takes the server runs in one batch: the engine's kAceSynthMaxTakes
// (pipeline-synth.h). Its copy here goes once the server refuses a batch
// past it itself (cadenza-audio #1182, item 5).
export const SERVER_MAX_TAKES = 9;

// a /synth batch: what is sent, and the takes it makes, one per seed.
export interface SynthBatch {
	toSend: AceRequest[];
	expanded: AceRequest[];
}

// the engine's row for a task, from the list /props gives; undefined for
// one it does not list, or before /props has loaded (cadenza-audio #1124)
export function taskRow(
	tasks: readonly AceTaskInfo[] | undefined,
	name: string
): AceTaskInfo | undefined {
	return tasks?.find((t) => t.name === name);
}

// the part of a region the source's waveform can show, in seconds, or null
// for none: its end read as the engine reads it (asked_region_end) -- a
// negative end is the source's end, or for a start before the source its
// start -- and the whole cut to the source's [0, duration]. So a repaint or
// a lego from a start to -1 shows to the source's end, and an outpaint
// before the source shows only its part on the source (cadenza-audio #1189).
export function regionOnSource(
	start: number,
	end: number,
	duration: number
): [number, number] | null {
	const asked = end >= 0 ? end : start < 0 ? 0 : duration;
	const from = Math.max(0, start);
	const to = Math.min(duration, asked);
	return to > from ? [from, to] : null;
}

// build a /synth batch from the pending requests and the form, or say why
// it cannot be sent (cadenza-audio #1175). Every request takes the form's
// synth sections (batchRequest); each its own seed, `seed()` when the form
// sets none, and the form's batch size, which the server expands to
// seed, seed+1, ... and `expanded` predicts for the take list. An adapter
// the server does not list is dropped. A text2music batch whose requests
// mix audio codes and none is refused, and so are codes that hold no code,
// both as the server refuses them (cadenza-audio #1159) -- for a task the
// engine's table says takes no source, by the rows /props lists; before they
// have loaded the server says it (#1124); so is a batch past the server's 9
// takes, which it would cut short unsaid.
export function buildSynthBatch(
	pending: AceRequest[],
	form: AceRequest,
	adapters: readonly string[],
	seed: () => number,
	tasks?: readonly AceTaskInfo[]
): SynthBatch | string {
	// whole, as the server reads it, and 1 at least
	const synthBatch = Math.max(1, whole(num(form.synth_batch_size)) ?? 1);
	// whole, as the server reads it, so the take list says the seeds it ran;
	// a negative one is a random seed
	const userSeed = whole(num(form.seed));
	const hasSeed = userSeed != null && userSeed >= 0;
	const shared = pickSections(form, SYNTH_SECTIONS);
	// seed and synth_batch_size are per request, set below
	delete shared.seed;
	delete shared.synth_batch_size;
	if (shared.adapter && !adapters.includes(String(shared.adapter))) delete shared.adapter;

	// the server's rules, as it reads codes: given when not empty, and then
	// holding one (holdsCode)
	const task = String(shared.task_type || 'text2music');
	const codes = pending.map(codesOf);
	const given = codes.map((c) => c !== '');
	const row = taskRow(tasks, task);
	if (row && !row.needs_source) {
		if (codes.some((c, i) => given[i] && !holdsCode(c))) {
			return "A composition's audio codes hold no code: fix them, or clear them.";
		}
		if (given.some(Boolean) && !given.every(Boolean)) {
			return 'Some compositions carry audio codes and some do not: give every one its codes, or clear them all.';
		}
	}
	if (pending.length * synthBatch > SERVER_MAX_TAKES) {
		return `A batch makes ${SERVER_MAX_TAKES} takes at most, and this asks for ${pending.length * synthBatch}: lower Batch, or synthesize fewer compositions.`;
	}
	// a Seed whose last take passes the engine's range would run as another
	// seed's twin (or, past a double's, be sent as null): refused. A random
	// seed is held below it, so its takes stay in range too.
	const lastSeed = SEED_MAX - (synthBatch - 1);
	if (hasSeed && (userSeed as number) > lastSeed) {
		return `A Seed runs its takes from it up: give one from 0 to ${lastSeed}, or none for a random seed.`;
	}

	const toSend: AceRequest[] = [];
	const expanded: AceRequest[] = [];
	for (const [i, p] of pending.entries()) {
		const r = batchRequest(p, shared);
		if ('audio_codes' in r) r.audio_codes = codes[i];
		const base = hasSeed ? (userSeed as number) : Math.min(seed(), lastSeed);
		toSend.push({ ...r, seed: base, synth_batch_size: synthBatch });
		// each take's song keeps the batch size it was made at, so loading
		// it sets the toolbar's again
		for (let take = 0; take < synthBatch; take++) {
			expanded.push({ ...r, seed: base + take, synth_batch_size: synthBatch });
		}
	}
	return { toSend, expanded };
}

// extract non-null fields from specified sections.
// caller removes per-expansion fields (seed, synth_batch_size) and injects srcRange.
export function pickSections(
	r: AceRequest,
	sections: readonly FieldSection[]
): Partial<AceRequest> {
	const out: Partial<AceRequest> = {};
	const allow = new Set(sections);
	for (const f of FIELDS) {
		if (!allow.has(f.section)) continue;
		const val = resolveField(f, get(r, f.key));
		if (val !== undefined) set(out, f.key, val);
	}
	return out;
}
