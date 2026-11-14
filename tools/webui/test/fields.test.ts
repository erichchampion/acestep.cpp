// A /synth batch as synthesize() builds it (cadenza-audio #1164, #1175):
// every request takes the form's synth sections, a field cleared after
// Compose included, its own seed and the form's batch size; a text2music
// batch mixing audio codes and none is refused before it is sent. Run with
// Node's own TypeScript support, no packages: `node tools/webui/test/fields.test.ts`
// (ctest test-webui-fields). Outside src/, so svelte-check does not type it;
// it imports fields.ts directly, which works while fields.ts imports only
// types.

import {
	buildSynthBatch,
	pickSections,
	SYNTH_SECTIONS,
	type SynthBatch
} from '../src/lib/fields.ts';
import type { AceRequest } from '../src/lib/types.ts';

let failed = 0;
function expect(what: string, ok: boolean, got?: unknown): void {
	if (!ok) {
		failed++;
		console.error(`FAIL: ${what}${got === undefined ? '' : `: ${JSON.stringify(got)}`}`);
	}
}
function built(b: SynthBatch | string): SynthBatch {
	if (typeof b === 'string') throw new Error(`refused: ${b}`);
	return b;
}
let next = 100;
const seed = () => (next += 1000);

// The real case: Compose made two; request 0 is savePending's copy of the
// form, Steps since cleared, and request 1 the LM's echo, Steps 8 and a
// track. The form sets a solver, an adapter the server lists, and Batch 2.
const pending = [
	{ caption: 'one', lyrics: 'a', bpm: 120 },
	{ caption: 'two', lyrics: 'b', bpm: 96, inference_steps: 8, track: 'guitar', lm_seed: 7 }
] as AceRequest[];
const form = {
	caption: 'one',
	solver: 'euler',
	track: '',
	adapter: 'warm.gguf',
	synth_batch_size: 2
} as AceRequest;
const { toSend, expanded } = built(buildSynthBatch(pending, form, ['warm.gguf'], seed));

const synthOf = (r: AceRequest) => {
	const s = pickSections(r, SYNTH_SECTIONS);
	delete s.seed;
	return JSON.stringify(s);
};
expect(
	'the requests agree in every synth field',
	synthOf(toSend[0]) === synthOf(toSend[1]),
	toSend
);
for (const [i, r] of toSend.entries()) {
	expect(`request ${i} keeps no cleared Steps`, !('inference_steps' in r), r);
	expect(`request ${i} keeps no cleared track`, !('track' in r), r);
	expect(
		`request ${i} has the form's solver and adapter`,
		r.solver === 'euler' && r.adapter === 'warm.gguf',
		r
	);
	expect(`request ${i} asks the form's batch`, r.synth_batch_size === 2, r);
}
expect(
	"each request's content and metadata stay its own",
	toSend[1].lyrics === 'b' && toSend[1].bpm === 96
);
expect('a per-item LM seed stays', toSend[1].lm_seed === 7, toSend[1]);
expect('each request its own seed', toSend[0].seed === 1100 && toSend[1].seed === 2100, toSend);
expect(
	'the takes are each seed and the next',
	expanded.map((r) => r.seed).join() === '1100,1101,2100,2101'
);
expect(
	'each take keeps the batch size',
	expanded.every((r) => r.synth_batch_size === 2),
	expanded
);
expect('the pending request is not changed', pending[1].inference_steps === 8, pending[1]);

// The form's seed, when set, is every request's base; an adapter the
// server does not list is dropped.
const seeded = built(
	buildSynthBatch(pending, { ...form, seed: 5, adapter: 'gone.gguf' }, [], seed)
);
expect(
	"the form's seed for each",
	seeded.toSend.every((r) => r.seed === 5),
	seeded.toSend
);
expect(
	'an unlisted adapter dropped',
	seeded.toSend.every((r) => !('adapter' in r)),
	seeded.toSend
);

// Codes: all or none run; a text2music mix is refused, by a message.
const codes = (c: string[]) =>
	c.map((a, i) => ({ caption: `c${i}`, audio_codes: a }) as AceRequest);
expect(
	'codes in all run',
	typeof buildSynthBatch(codes(['1,2', '3']), form, [], seed) !== 'string'
);
expect('codes in none run', typeof buildSynthBatch(codes(['', ' ']), form, [], seed) !== 'string');
const mixed = buildSynthBatch(codes(['1,2', '']), form, [], seed);
expect('a mix is refused', typeof mixed === 'string' && mixed.includes('audio codes'), mixed);
const cover = buildSynthBatch(codes(['1,2', '']), { ...form, task_type: 'cover' }, [], seed);
expect("a cover's codes are each request's own", typeof cover !== 'string', cover);

if (failed) {
	console.error(`${failed} failed`);
	process.exit(1);
}
console.log('fields: ok');
