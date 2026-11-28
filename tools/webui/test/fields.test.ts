// A /synth batch as synthesize() builds it (cadenza-audio #1164, #1175):
// every request takes the form's synth sections, a field cleared after
// Compose included, its own seed and the form's batch size; a text2music
// batch mixing audio codes and none is refused before it is sent. Run with
// Node's own TypeScript support, no packages: `node tools/webui/test/fields.test.ts`
// (ctest test-webui-fields). Outside src/, so svelte-check does not type it;
// it imports fields.ts directly, which works while fields.ts imports only
// types.

import {
	asSent,
	buildSparse,
	buildSynthBatch,
	SEED_MAX,
	codesOf,
	holdsCode,
	regionAfterDrag,
	regionOnSource,
	taskRow,
	pickSections,
	SYNTH_SECTIONS,
	type SynthBatch
} from '../src/lib/fields.ts';
import type { AceRequest, AceTaskInfo } from '../src/lib/types.ts';

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
// form, Steps since cleared and a track still set, and request 1 the LM's
// echo, Steps 8 and a track. The form sets a solver, an adapter the server
// lists, and Batch 2.
const pending = [
	{ caption: 'one', lyrics: 'a', bpm: 120, track: 'drums' },
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
const own =
	toSend[0].caption === 'one' &&
	toSend[0].bpm === 120 &&
	toSend[1].caption === 'two' &&
	toSend[1].lyrics === 'b' &&
	toSend[1].bpm === 96;
expect("each request's content and metadata stay its own", own, toSend);
expect('a per-item LM seed stays', toSend[1].lm_seed === 7, toSend[1]);
expect('each request its own seed', toSend[0].seed === 1100 && toSend[1].seed === 2100, toSend);
const seeds = expanded.map((r) => r.seed).join();
expect('the takes are each seed and the next', seeds === '1100,1101,2100,2101', seeds);
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

// Codes, as the server reads them: all or none run; a text2music mix is
// refused, and so are codes that hold no code -- white space or commas
// alone; a cover's codes are each request's own.
const codes = (c: string[]) =>
	c.map((a, i) => ({ caption: `c${i}`, audio_codes: a }) as AceRequest);
// The task rows /props lists (#1124): the codes rule is a task's that takes
// no source, by its row; with no rows loaded the server says it.
const row = (name: string, needs_source: boolean): AceTaskInfo => ({
	name,
	needs_source,
	needs_caption: true,
	cover_knobs: false,
	reads_region: false,
	reads_track: false,
	splices: false
});
const tasks = [row('text2music', false), row('cover', true), row('a-new-task', false)];
expect('a row is found by name', taskRow(tasks, 'cover')?.needs_source === true);
expect('no row for a task not listed', taskRow(tasks, 'covers') === undefined);
expect('no row before /props loads', taskRow(undefined, 'cover') === undefined);
const sourceless = (task: string, c: string[]) =>
	buildSynthBatch(codes(c), { ...form, task_type: task }, [], seed, tasks);
const unloaded = buildSynthBatch(codes(['1,2', '']), form, [], seed);
expect("with no rows, a mix is the server's to refuse", typeof unloaded !== 'string', unloaded);
const newTask = sourceless('a-new-task', ['1,2', '']);
expect(
	'a new sourceless task the engine lists is held to the codes rule',
	typeof newTask === 'string' && newTask.includes('some do not'),
	newTask
);

// The region the source's waveform shows, its -1 end read as the engine
// reads it (asked_region_end, #1189).
const same = (a: [number, number] | null, b: [number, number] | null) =>
	a === b || (a !== null && b !== null && a[0] === b[0] && a[1] === b[1]);
expect('a region of its own', same(regionOnSource(2, 5, 10), [2, 5]));
expect('a -1 end is the source end', same(regionOnSource(2, -1, 10), [2, 10]));
expect('an end past the source is cut to it', same(regionOnSource(2, 12, 10), [2, 10]));
expect(
	'an outpaint before the source shows its part on it',
	same(regionOnSource(-3, 4, 10), [0, 4])
);
expect('an outpaint to -1 shows nothing on the source', same(regionOnSource(-3, -1, 10), null));
expect('an end at or before the start shows nothing', same(regionOnSource(5, 5, 10), null));
expect('a region past the source shows nothing', same(regionOnSource(11, 12, 10), null));

// A drag's edit of the region (#1189, #1182): only the edge moved is
// written; a range drawn empty mid-drag writes nothing; one that ends empty
// -- a click with no drag, a handle dropped on the other -- clears it.
const asked = { start: -3, end: -1 };
const region = (r: { start: number | null; end: number | null }) => `${r.start},${r.end}`;
expect(
	'moving the start keeps the asked end',
	region(regionAfterDrag('lo', 2, 10, asked)) === '2,-1'
);
expect(
	'moving the end keeps the asked start',
	region(regionAfterDrag('hi', 0, 4, asked)) === '-3,4'
);
expect('a new range writes both', region(regionAfterDrag('new', 1, 3, asked)) === '1,3');
expect(
	'a range drawn empty mid-drag writes nothing',
	region(regionAfterDrag('new', 2, 2, asked)) === '-3,-1'
);
expect(
	'a handle on the other mid-drag writes nothing',
	region(regionAfterDrag('lo', 4, 4, asked)) === '-3,-1'
);
expect(
	'a drag that ends empty clears the region',
	region(regionAfterDrag('clear', 2, 2, asked)) === 'null,null'
);
expect(
	'clearing a region never asked is still none',
	region(regionAfterDrag('clear', 0, 0, { start: null, end: null })) === 'null,null'
);

const all = buildSynthBatch(codes(['1,2', ' ,3']), form, [], seed, tasks);
expect('codes in all run', typeof all !== 'string', all);
const none = buildSynthBatch(codes(['', '']), form, [], seed, tasks);
expect('codes in none run', typeof none !== 'string', none);
const mixed = buildSynthBatch(codes(['1,2', '']), form, [], seed, tasks);
expect('a mix is refused', typeof mixed === 'string' && mixed.includes('some do not'), mixed);
// The white space a paste leaves around codes is trimmed, as the form's
// light reads it (#1182): a space or a newline alone is no codes, and codes
// in a newline are sent without it.
const blank = buildSynthBatch(codes(['', ' ']), form, [], seed, tasks);
expect('codes of a space alone are none', typeof blank !== 'string', blank);
const pasted = buildSynthBatch(codes(['\n1,2\n', ' 3 ']), form, [], seed, tasks);
expect(
	'pasted codes are sent trimmed',
	typeof pasted !== 'string' && pasted.toSend.map((r) => r.audio_codes).join('|') === '1,2|3',
	pasted
);
const strayNewline = buildSynthBatch(codes(['1,2', '\n']), form, [], seed, tasks);
expect(
	'a newline alone is no codes, so beside codes it is a mix',
	typeof strayNewline === 'string' && strayNewline.includes('some do not'),
	strayNewline
);
expect("codesOf trims as it's sent", codesOf({ audio_codes: '\t 5,6 \n' } as AceRequest) === '5,6');
// Codes wrapped over lines are sent whole: the server stops at a break.
const wrapped = buildSynthBatch(codes(['1,2,3\n4,5,6', '7\t8']), form, [], seed, tasks);
expect(
	'codes over lines are sent whole',
	typeof wrapped !== 'string' &&
		wrapped.toSend.map((r) => r.audio_codes).join('|') === '1,2,3 4,5,6|7 8',
	wrapped
);
expect(
	'an export carries the codes as sent',
	asSent({ caption: 'x', audio_codes: '\n1,2\n3\n' } as AceRequest).audio_codes === '1,2 3'
);
expect(
	'an export of a newline alone carries no codes',
	!('audio_codes' in asSent({ caption: 'x', audio_codes: '\n' } as AceRequest))
);
expect(
	'a pending request keeps its codes as typed',
	buildSparse({ caption: 'x', audio_codes: '1,2\n3' } as AceRequest).audio_codes === '1,2\n3'
);
const commas = buildSynthBatch(codes([',', '']), form, [], seed, tasks);
expect(
	'codes of a comma hold no code',
	typeof commas === 'string' && commas.includes('no code'),
	commas
);
// A code is an int, as from_chars reads it (ops_codes_given, #1182): a run of
// digits past an int's range holds none.
expect('the largest int is a code', holdsCode('2147483647'));
expect('the smallest int is a code', holdsCode(', -2147483648'));
expect('one past the largest int is none', !holdsCode('4294967296'));
expect('one past the smallest int is none', !holdsCode('-2147483649'));
expect('a sign alone is none', !holdsCode('-,1'));
const tooBig = buildSynthBatch(codes(['4294967296', '1']), form, [], seed, tasks);
expect(
	'codes past an int are refused as holding none',
	typeof tooBig === 'string' && tooBig.includes('no code'),
	tooBig
);
const cover = buildSynthBatch(codes(['1,2', '']), { ...form, task_type: 'cover' }, [], seed, tasks);
expect("a cover's codes are each request's own", typeof cover !== 'string', cover);

// The server's 9 takes: 4 compositions at Batch 3 are refused, not cut to
// 9; a Batch of 2.5 is 2, as the server reads it.
const many = buildSynthBatch(
	codes(['', '', '', '']),
	{ ...form, synth_batch_size: 3 },
	[],
	seed,
	tasks
);
expect('past 9 takes is refused', typeof many === 'string' && many.includes('9 takes'), many);
const half = built(buildSynthBatch(pending, { ...form, synth_batch_size: 2.5 }, [], seed));
expect(
	'a fractional Batch is whole',
	half.expanded.length === 4 && half.toSend[0].synth_batch_size === 2,
	half
);
// Exactly 9 takes run (#1182): 3 compositions at Batch 3.
const nine = buildSynthBatch(
	codes(['', '', '']),
	{ ...form, synth_batch_size: 3 },
	[],
	seed,
	tasks
);
expect('exactly 9 takes run', typeof nine !== 'string' && nine.expanded.length === 9, nine);
// A Batch below 1, or none, is 1: one take a composition.
for (const below of [0, -2, 0.5, '', undefined]) {
	const b = buildSynthBatch(pending, { ...form, synth_batch_size: below as number }, [], seed);
	expect(
		`a Batch of ${JSON.stringify(below)} is 1`,
		typeof b !== 'string' &&
			b.expanded.length === 2 &&
			b.toSend.every((r) => r.synth_batch_size === 1),
		b
	);
}
// A Seed's takes run from it up, in the engine's [0, UINT32_MAX] (request.h):
// one whose last take passes it is refused -- Infinity too, which JSON would
// send as null -- and a random seed is held below it.
const atBatch2 = (s: number) =>
	buildSynthBatch(pending, { ...form, seed: s, synth_batch_size: 2 }, [], seed);
expect('a Seed whose last take is the most runs', typeof atBatch2(SEED_MAX - 1) !== 'string');
for (const big of [SEED_MAX, 2 ** 53, 1e400]) {
	const b = atBatch2(big);
	expect(`a Seed of ${big} at Batch 2 is refused`, typeof b === 'string' && b.includes('0 to'), b);
}
for (const negative of [-1, -1e400, ' ']) {
	const b = built(atBatch2(negative as number));
	expect(
		`a Seed of ${JSON.stringify(negative)} is a random one`,
		b.toSend.every((r) => (r.seed as number) >= 0 && (r.seed as number) < SEED_MAX),
		b
	);
}
const topRandom = built(
	buildSynthBatch(pending, { ...form, synth_batch_size: 2 }, [], () => SEED_MAX)
);
expect(
	"a random seed's takes stay in range",
	topRandom.expanded.every((r) => (r.seed as number) <= SEED_MAX),
	topRandom.expanded
);
// The cap is said before the Seed: a Batch past it is the cap's to refuse.
const hugeBatch = buildSynthBatch(pending, { ...form, seed: 5, synth_batch_size: 1e20 }, [], seed);
expect(
	'a huge Batch is refused by the cap, not the Seed',
	typeof hugeBatch === 'string' && hugeBatch.includes('takes at most'),
	hugeBatch
);
// A fractional Seed is whole, as the server reads it (toward zero), so the
// take list says the seeds that ran.
const fractional = built(buildSynthBatch(pending, { ...form, seed: 5.5 }, [], seed));
expect(
	'a fractional Seed is whole',
	fractional.toSend.every((r) => r.seed === 5) &&
		fractional.expanded.map((r) => r.seed).join() === '5,6,5,6',
	fractional
);

if (failed) {
	console.error(`${failed} failed`);
	process.exit(1);
}
console.log('fields: ok');
