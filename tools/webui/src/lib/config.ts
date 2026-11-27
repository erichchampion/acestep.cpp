// UI constants
export const PROPS_POLL_MS = 10000;
export const FETCH_TIMEOUT_MS = 2000;
export const JOB_POLL_MS = 2000;
export const SSE_RECONNECT_MS = 2000;
export const LOG_MAX_LINES = 50;
export const WAVEFORM_HEIGHT = 64;
export const WAVEFORM_BINS = 4096;

// The words the form shows for each task. The tasks themselves, and what
// each reads, are the engine's: GET /props lists them from its task table
// (cadenza-audio #1124), and the form offers the ones it lists, in its
// order. A listed task with no words here shows its name; before /props has
// loaded, the form offers these.
export const TASK_LABELS: Readonly<Record<string, string>> = {
	text2music: 'Text2Music: from prompt and LM codes',
	cover: 'Cover: reinterpret in a new style',
	'cover-nofsq': 'Cover (no FSQ): closer to the original',
	repaint: 'Repaint: regenerate a region',
	lego: 'Lego: add a stem over backing audio',
	extract: 'Extract: isolate one stem from a mix',
	complete: 'Complete: auto-arrange around a partial track'
};

// The tasks whose track is one stem, so the form offers one track pill at a
// time: the form's choice of control, which the engine's table has no
// column for (a complete takes several).
export const SINGLE_TRACK_TASKS: readonly string[] = ['lego', 'extract'];

// Solver names (mirrors task-types.h SOLVER_* and the C++ solver registry).
// The string is the canonical solver key resolved by solver_lookup().
export const SOLVER_EULER = 'euler';
export const SOLVER_SDE = 'sde';
export const SOLVER_DPM3M = 'dpm3m';
export const SOLVER_STORK4 = 'stork4';

// DCW modes (mirrors task-types.h DCW_MODE_*)
export const DCW_MODE_LOW = 'low';
export const DCW_MODE_HIGH = 'high';
export const DCW_MODE_DOUBLE = 'double';
export const DCW_MODE_PIX = 'pix';

export const TRACK_NAMES = [
	'vocals',
	'backing_vocals',
	'drums',
	'bass',
	'guitar',
	'keyboard',
	'percussion',
	'strings',
	'synth',
	'fx',
	'brass',
	'woodwinds'
] as const;
