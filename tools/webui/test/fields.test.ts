// The form's synth sections are the batch's (cadenza-audio #1164): a field
// cleared after Compose is gone from every request, not kept from the copy
// the LM echoed into a pending one. Run with Node's own TypeScript support,
// no packages: `node tools/webui/test/fields.test.ts` (ctest
// test-webui-fields). Outside src/, so svelte-check does not type it.

import { pickSections, withoutSections, type FieldSection } from '../src/lib/fields.ts';
import type { AceRequest } from '../src/lib/types.ts';

let failed = 0;
function expect(what: string, ok: boolean, got?: unknown): void {
	if (!ok) {
		failed++;
		console.error(`FAIL: ${what}${got === undefined ? '' : `: ${JSON.stringify(got)}`}`);
	}
}

const synth: FieldSection[] = ['flow', 'advanced', 'toolbar', 'routing'];
// Two pending requests from a Compose with Steps 8, the LM's echo of it, and
// a track; the form since cleared Steps and the track, and set a solver.
const pending = [
	{ caption: 'one', lyrics: 'a', bpm: 120, inference_steps: 8, track: 'guitar' },
	{ caption: 'two', lyrics: 'b', bpm: 96, inference_steps: 8, track: 'guitar', lm_seed: 7 }
] as AceRequest[];
const form = { caption: 'one', solver: 'euler', track: '' } as AceRequest;
const params = pickSections(form, synth);
const sent = pending.map((p) => ({ ...withoutSections(p, synth), ...params }) as AceRequest);

for (const [i, r] of sent.entries()) {
	expect(`request ${i} keeps no cleared Steps`, !('inference_steps' in r), r);
	expect(`request ${i} keeps no cleared track`, !('track' in r), r);
	expect(`request ${i} has the form's solver`, r.solver === 'euler', r);
}
expect(
	"content and metadata stay each request's own",
	sent[0].caption === 'one' && sent[1].caption === 'two' && sent[1].bpm === 96 && sent[1].lyrics === 'b',
	sent
);
expect('a per-item LM seed stays', sent[1].lm_seed === 7, sent[1]);
expect(
	'the requests agree in every synth field',
	JSON.stringify(pickSections(sent[0], synth)) === JSON.stringify(pickSections(sent[1], synth)),
	sent
);
expect('the pending request is not changed', pending[0].inference_steps === 8, pending[0]);

if (failed) {
	console.error(`${failed} failed`);
	process.exit(1);
}
console.log('fields: ok');
