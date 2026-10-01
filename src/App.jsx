import React, { useState, useEffect, useMemo, useRef, useContext, createContext } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, LabelList, LineChart, Line
} from 'recharts';
import {
  Play, Pause, SkipBack, SkipForward, RotateCcw, Sparkles, ArrowRight, Wand2, Shield, Lock, Unlock
} from 'lucide-react';

/* ============================== DATA / LOGIC ============================== */

const CATEGORY_WORDS = {
  animal: ['cat', 'cats', 'dog', 'dogs', 'mouse', 'mice', 'bird', 'birds', 'lion', 'tiger', 'puppy', 'puppies',
    'kitten', 'kittens', 'animal', 'animals'],
  action: ['sat', 'sit', 'sits', 'ran', 'run', 'runs', 'jumped', 'jump', 'ate', 'eat', 'eats', 'sleeps', 'slept',
    'sleep', 'played', 'play', 'loves', 'love', 'loved', 'sings', 'sing', 'barks', 'bark', 'meowed', 'meow',
    'walked', 'walk', 'walks', 'opened', 'open'],
  object: ['mat', 'floor', 'park', 'house', 'table', 'pizza', 'bed', 'chair', 'tree', 'ball', 'book', 'door',
    'gate', 'couch', 'rug', 'moon', 'banana', 'coffee', 'music', 'coding', 'mondays', 'yard', 'garden', 'window',
    'box', 'car', 'cars', 'truck', 'trucks', 'bike', 'bikes', 'bus', 'train', 'plane', 'boat', 'vehicle', 'vehicles'],
  function: ['the', 'a', 'an', 'on', 'in', 'at', 'is', 'was', 'were', 'are', 'to', 'of', 'and', 'it', 'this', 'that',
    'i', 'you', 'he', 'she', 'they', 'we', 'with', 'for', 'while', 'because', 'however', 'which', 'so', 'but', 'near']
};

function categoryOf(word) {
  const w = word.toLowerCase().replace(/[^a-z']/g, '');
  for (const [cat, list] of Object.entries(CATEGORY_WORDS)) {
    if (list.includes(w)) return cat;
  }
  return 'misc';
}

const CATEGORY_COLOR = { animal: '#F2A93B', action: '#FB7185', object: '#5EEAD4', function: '#64748B', misc: '#A78BFA' };
const CATEGORY_NAME = { animal: 'animal / subject', action: 'action / verb', object: 'object / place', function: 'function word', misc: 'other' };
const CATEGORY_BASE = {
  animal: { x: 24, y: 26 }, action: { x: 76, y: 32 }, object: { x: 68, y: 80 },
  function: { x: 20, y: 78 }, misc: { x: 48, y: 52 }
};

/* ---- Glossary: click-to-define tooltips so every term is explained in place ---- */

const GLOSSARY = {
  token: "A small chunk of text — often a whole word, sometimes part of one — that the model treats as a single unit.",
  tokenId: 'The number each token gets mapped to. Models only do math on numbers, never on raw letters.',
  embedding: "A list of numbers that represents a token's meaning. Words with similar meaning end up with similar numbers.",
  positionalEncoding: "A unique numeric pattern added to each token showing where it sits in the sentence — without this, word order would be lost.",
  query: "What a token is 'searching for' when it looks at other tokens — compared against everyone else's Key.",
  key: "What a token 'advertises' about itself — compared against another token's Query to judge relevance.",
  value: 'The actual content a token passes along once another token decides it is relevant.',
  softmax: 'A function that turns raw scores into percentages that always add up to 100%, so they can be read as probabilities.',
  causalMasking: 'A rule that blocks a token from looking at words that come after it — the reason models can only predict left to right.',
  multiHead: "Running several attention 'lenses' side by side, each learning a different kind of relationship, then combining the results.",
  addNorm: "Two things done after every sub-layer: 'Add' feeds the input back in (a shortcut so training signals do not get lost), and 'Norm' rescales the numbers to a stable range.",
  feedForward: 'A small 2-step network applied to each token on its own (using ReLU: keep positive numbers, zero out negatives), adding processing after attention.',
  confidence: "How strongly the model favors its top choice right now — higher means it's more sure.",
  probability: 'A percentage-style score showing how likely the model thinks an option is. All the options shown together add up to 100%.',
  repetitionPenalty: 'A rule that lowers the score of words just used, so the model does not get stuck repeating itself.',
  vocabulary: 'The full list of tokens a model can choose from — real models pick from 50,000+, this demo shows a handful for clarity.',
  decoderOnly: "A design that uses only the 'decoder' half of the original Transformer, generating text one word at a time, left to right — what GPT-style LLMs use."
};

const TermContext = createContext({ openId: null, setOpenId: () => {} });

function Term({ id, children }) {
  const { openId, setOpenId } = useContext(TermContext);
  const isOpen = openId === id;
  return (
    <span className="lpe-term-wrap">
      <button type="button" className="lpe-term-btn" onClick={(e) => { e.stopPropagation(); setOpenId(isOpen ? null : id); }}>
        {children}<span className="lpe-term-i">ⓘ</span>
      </button>
      {isOpen && <span className="lpe-term-pop" onClick={(e) => e.stopPropagation()}>{GLOSSARY[id]}</span>}
    </span>
  );
}

function hslColor(i, total) {
  const hue = Math.round((360 / Math.max(total, 1)) * i);
  return `hsl(${hue}, 70%, 62%)`;
}

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) { h = (h << 5) - h + s.charCodeAt(i); h |= 0; }
  return Math.abs(h);
}

function tokenize(sentence) {
  return sentence.trim().split(/\s+/).filter(Boolean).map((w) => w.replace(/[.,!?;:]+$/, ''));
}
function tokenId(word) { return 100 + (hashStr(word.toLowerCase()) % 9899); }

function embedPosition(word) {
  const cat = categoryOf(word);
  const base = CATEGORY_BASE[cat];
  const h = hashStr(word.toLowerCase());
  const jx = ((h % 100) / 100 - 0.5) * 16;
  const jy = (((h >> 8) % 100) / 100 - 0.5) * 16;
  return { x: Math.max(4, Math.min(96, base.x + jx)), y: Math.max(4, Math.min(96, base.y + jy)), category: cat };
}

/* ---- Self-attention (causal / masked, like real decoder-only LLMs) ---- */

function computeAttention(tokens, focusIndex, sharpness = 1, causal = true) {
  const catI = categoryOf(tokens[focusIndex]);
  const raw = tokens.map((tok, j) => {
    if (j === focusIndex) return 0.42;
    const catJ = categoryOf(tok);
    let base = 0.12;
    if (catJ === 'function') base = 0.05;
    if (catI === 'action' && catJ === 'animal') base = 0.85;
    if (catI === 'action' && catJ === 'object') base = 0.55;
    if ((catI === 'animal' || catI === 'object') && catJ === 'action') base = 0.35;
    const dist = Math.abs(focusIndex - j);
    base *= 1 / (1 + 0.25 * dist);
    return base;
  });
  const masked = raw.map((v, j) => (causal && j > focusIndex ? 1e-6 : v));
  const sharpened = masked.map((v) => Math.pow(v, sharpness));
  const sum = sharpened.reduce((a, b) => a + b, 0) || 1;
  return sharpened.map((v) => v / sum);
}

/* ---- Simplified multi-head attention (paper uses h=8; we show 4 for clarity) ---- */

const HEAD_META = [
  { label: 'Head 1', role: 'syntactic — verb ↔ subject/object', color: '#F2A93B' },
  { label: 'Head 2', role: 'positional — nearby words', color: '#5EEAD4' },
  { label: 'Head 3', role: 'semantic — same word type', color: '#FB7185' },
  { label: 'Head 4', role: 'contextual — broad + function words', color: '#A78BFA' }
];

function computeAttentionHead(tokens, focusIndex, headIdx, causal) {
  const catI = categoryOf(tokens[focusIndex]);
  const raw = tokens.map((tok, j) => {
    if (j === focusIndex) return 0.35;
    const catJ = categoryOf(tok);
    const dist = Math.abs(focusIndex - j);
    let base;
    if (headIdx === 0) {
      base = 0.1;
      if (catI === 'action' && catJ === 'animal') base = 0.8;
      else if (catI === 'action' && catJ === 'object') base = 0.5;
      else if ((catI === 'animal' || catI === 'object') && catJ === 'action') base = 0.4;
      base *= 1 / (1 + 0.2 * dist);
    } else if (headIdx === 1) {
      base = 1 / (1 + 0.6 * dist);
    } else if (headIdx === 2) {
      base = catJ === catI ? 0.6 : 0.08;
      base *= 1 / (1 + 0.3 * dist);
    } else {
      base = catJ === 'function' ? 0.25 : 0.15;
      base *= 1 / (1 + 0.1 * dist);
    }
    return base;
  });
  const masked = raw.map((v, j) => (causal && j > focusIndex ? 1e-6 : v));
  const sum = masked.reduce((a, b) => a + b, 0) || 1;
  return masked.map((v) => v / sum);
}

/* ---- Sinusoidal positional encoding, straight from the paper's formula ---- */

function positionalEncodingSample(pos, dims = 16, dModel = 64) {
  const arr = [];
  for (let i = 0; i < dims; i++) {
    const denom = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
    arr.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom));
  }
  return arr;
}

/* ---- Next-word prediction: rule-based + repetition penalty (fixes "it it it" loops) ---- */

const KNOWN_CONTINUATIONS = {
  'the cat sat on the': [['mat', 0.72], ['floor', 0.14], ['rug', 0.06], ['chair', 0.04], ['couch', 0.03], ['moon', 0.01]],
  'i love': [['pizza', 0.34], ['you', 0.24], ['music', 0.16], ['coding', 0.12], ['coffee', 0.09], ['mondays', 0.05]],
  'the dog ran to the': [['park', 0.48], ['door', 0.22], ['gate', 0.16], ['tree', 0.09], ['moon', 0.05]],
  'she opened the': [['door', 0.5], ['book', 0.22], ['box', 0.16], ['window', 0.12]]
};

const PREPOSITIONS = ['on', 'in', 'at', 'to', 'with', 'for', 'near', 'of'];
const ARTICLES = ['the', 'a', 'an'];
const COPULAS = ['is', 'are', 'was', 'were'];

function weightedList(words, seed) {
  const uniq = [...new Set(words)].slice(0, 6);
  if (uniq.length === 0) return [['something', 1]];
  return uniq.map((w, i) => [w, (1 / (i + 1)) * (0.7 + 0.6 * (((seed >> (i * 4)) & 0xff) / 255))]);
}

function predictNext(tokens) {
  if (!tokens || tokens.length === 0) return [['the', 0.3], ['i', 0.25], ['a', 0.2], ['it', 0.15], ['this', 0.1]];
  const clean = tokens.map((t) => t.toLowerCase());
  const key = clean.join(' ');

  for (const k of Object.keys(KNOWN_CONTINUATIONS)) {
    if (key === k || key.endsWith(' ' + k)) return applyRepetitionPenalty(KNOWN_CONTINUATIONS[k], clean);
  }

  const n = clean.length;
  const last = clean[n - 1];
  const secondLast = clean[n - 2];
  const h = hashStr(key);
  let pool;

  if (secondLast !== undefined && last === secondLast) {
    pool = weightedList(['then', 'again', 'quickly', 'together', 'suddenly', 'afterward'], h);
  } else if (ARTICLES.includes(last)) {
    const usedNouns = clean.filter((t) => categoryOf(t) === 'animal' || categoryOf(t) === 'object');
    pool = weightedList([...new Set(usedNouns)].reverse().concat(['park', 'table', 'door', 'room', 'tree', 'friend']), h);
  } else if (last === 'and') {
    let prevWord = null;
    for (let i = n - 2; i >= 0; i--) { if (categoryOf(clean[i]) !== 'function') { prevWord = clean[i]; break; } }
    const cat = prevWord ? categoryOf(prevWord) : null;
    const options = cat && CATEGORY_WORDS[cat] ? CATEGORY_WORDS[cat].filter((w) => w !== prevWord) : ['then', 'later'];
    pool = weightedList(options, h);
  } else if (COPULAS.includes(last)) {
    pool = weightedList(['great', 'ready', 'here', 'different', 'important', 'common'], h);
  } else if (categoryOf(last) === 'action') {
    pool = weightedList(['home', 'outside', 'again', 'quickly', 'there', 'away'], h);
  } else if (PREPOSITIONS.includes(last)) {
    pool = weightedList(['the', 'a', 'their', 'its', 'you', 'me'], h);
  } else {
    pool = weightedList(['and', 'but', 'so', 'which', 'however', 'because'], h);
  }

  return applyRepetitionPenalty(pool, clean);
}

function applyRepetitionPenalty(pool, clean) {
  const last = clean[clean.length - 1];
  const secondLast = clean[clean.length - 2];
  const withPenalty = pool.map(([w, wt]) => {
    let weight = wt;
    if (w === last) weight *= 0.04;
    else if (w === secondLast) weight *= 0.25;
    return [w, weight];
  });
  const sum = withPenalty.reduce((a, [, w]) => a + w, 0) || 1;
  return withPenalty.map(([w, wt]) => [w, wt / sum]).sort((a, b) => b[1] - a[1]).slice(0, 6);
}

const DEFAULT_SENTENCE = 'The cat sat on the';
const PRESETS = ['The cat sat on the mat', 'I love pizza', 'The dog ran to the park', 'She opened the door'];
const TOKEN_W = 90;
const N_LAYERS = 6; // matches the paper's base model (N=6)

const PAPER_SPECS = [
  { k: 'Layers (N)', v: '6' },
  { k: 'Model dim (d_model)', v: '512' },
  { k: 'Attention heads (h)', v: '8' },
  { k: 'Key/Value dim (d_k, d_v)', v: '64 each' },
  { k: 'Feed-forward dim (d_ff)', v: '2048' },
  { k: 'Dropout', v: '0.1' }
];

const STEPS = [
  { id: 'architecture', num: 1, label: 'Architecture' },
  { id: 'tokenize', num: 2, label: 'Tokenize' },
  { id: 'embed', num: 3, label: 'Embed' },
  { id: 'position', num: 4, label: 'Position' },
  { id: 'attention', num: 5, label: 'Attention' },
  { id: 'layers', num: 6, label: 'Layers' },
  { id: 'output', num: 7, label: 'Predict' },
  { id: 'pipeline', num: 8, label: 'Full pipeline' }
];

const PIPELINE_NODES = [
  { id: 'input', label: 'Your sentence' },
  { id: 'tokenize', label: 'Tokenize' },
  { id: 'embed', label: 'Embed' },
  { id: 'position', label: 'Position' },
  { id: 'attention', label: 'Attention' },
  { id: 'layers', label: 'Layers' },
  { id: 'output', label: 'Predict' }
];

function layerRoleText(layer, N) {
  const frac = layer / N;
  if (layer === N) return 'The final layer produces the sharpened, context-rich representation handed to the prediction step.';
  if (frac <= 0.34) return 'Early layers mostly track local, surface-level patterns — nearby words and basic grammatical roles.';
  if (frac <= 0.67) return 'Mid layers start linking related content words together, like connecting a verb to its likely subject.';
  return 'Deeper layers integrate broader sentence context, pulling in words further away when they matter.';
}

const STEP_INFO = {
  architecture: {
    title: 'Transformer Architecture',
    what: 'The overall stack of components that make up a Transformer-based language model.',
    why: 'Seeing the full architecture up front makes every later step click into place as "which box am I in right now."',
    use: 'This is the blueprint every one of the following steps is a zoomed-in view of.'
  },
  tokenize: {
    title: 'Tokenization',
    what: 'Splitting text into small units (tokens) and mapping each one to a number.',
    why: 'Neural networks only compute on numbers, not raw text — this is the doorway from language into math.',
    use: 'Always the first step of any pass through the model.'
  },
  embed: {
    title: 'Embeddings',
    what: 'Converting each token ID into a vector of numbers that encodes meaning.',
    why: 'A token ID alone is meaningless — similar-meaning words need to land near each other mathematically.',
    use: 'Gives the model a workable notion of "closeness in meaning" — these vectors later get projected into Query, Key and Value vectors for attention.'
  },
  position: {
    title: 'Positional Encoding',
    what: 'Adding a unique sine/cosine "fingerprint" to each token, based on its position.',
    why: 'Attention looks at all words at once, so word order would otherwise be lost.',
    use: 'Injected right before attention so sequence order survives into the next step.'
  },
  attention: {
    title: 'Self-Attention',
    what: 'Every token forms a Query, Key and Value; attention scores come from Query·Key, and the output is a weighted mix of Values.',
    why: 'Words change meaning with context. In real decoder-only LLMs this is masked so a token can never look ahead at words it hasn\'t generated yet.',
    use: 'The mechanism that lets the model connect related words, however far apart they sit — toggle Causal/Full masking above to compare.'
  },
  layers: {
    title: 'Stacked Layers',
    what: 'Each layer repeats: masked multi-head attention → Add & Norm → feed-forward network → Add & Norm — stacked N times.',
    why: 'One pass only captures so much — stacking builds progressively deeper, sharper understanding.',
    use: 'Depth is what gives large models their capacity for nuance.'
  },
  output: {
    title: 'Next-Token Prediction',
    what: 'The final representation is converted into a probability for every possible next word.',
    why: 'All the understanding built up has to turn into one concrete decision: what word comes next.',
    use: 'The literal output of one forward pass — and the loop behind text generation.'
  },
  pipeline: {
    title: 'Full Pipeline',
    what: 'Every step above, connected, running on your sentence end to end.',
    why: 'Seeing the whole flow at once is how the individual pieces click into one system.',
    use: 'This is what happens, start to finish, every time a model produces one word.'
  }
};

/* ============================== SMALL UI PARTS ============================== */

function ExplanationPanel({ stepId, extra }) {
  const info = STEP_INFO[stepId];
  return (
    <div className="lpe-panel lpe-explain">
      <div className="lpe-explain-grid">
        <div><span className="lpe-explain-label">What it is</span><p>{info.what}</p></div>
        <div><span className="lpe-explain-label">Why it exists</span><p>{info.why}</p></div>
        <div><span className="lpe-explain-label">Use in the process</span><p>{info.use}</p></div>
      </div>
      {extra && <div className="lpe-live-note"><Sparkles size={14} /> <span>{extra}</span></div>}
    </div>
  );
}

function TokenChip({ word, id, color, faded }) {
  return (
    <div className="lpe-chip" style={{ borderColor: color, opacity: faded ? 0.4 : 1 }}>
      <span className="lpe-chip-word">{word}</span>
      {id !== undefined && <span className="lpe-chip-id" style={{ color }}>{id}</span>}
    </div>
  );
}

function EmptyState({ children }) {
  return <div className="lpe-empty">{children}</div>;
}

/* ============================== ARCHITECTURE VIEW ============================== */

function ArchitectureView() {
  return (
    <div className="lpe-stage-inner">
      <div className="lpe-row-label">Most of today's chat-style LLMs (GPT, Llama, and this demo) use a <Term id="decoderOnly">decoder-only</Term> Transformer — one stack, generating left to right</div>
      <div className="lpe-arch-stack">
        <div className="lpe-arch-box output">Output probabilities (Softmax)</div>
        <ArrowRight className="lpe-arch-arrow" size={16} />
        <div className="lpe-arch-box">Linear</div>
        <ArrowRight className="lpe-arch-arrow" size={16} />
        <div className="lpe-arch-repeat">
          <div className="lpe-arch-box ff">Feed-Forward (ReLU)</div>
          <div className="lpe-arch-box norm">Add & Norm</div>
          <div className="lpe-arch-box attn">Masked Multi-Head Self-Attention</div>
          <div className="lpe-arch-box norm">Add & Norm</div>
          <span className="lpe-arch-repeat-label">× N (6 in the paper)</span>
        </div>
        <ArrowRight className="lpe-arch-arrow" size={16} />
        <div className="lpe-arch-box">Input Embedding + Positional Encoding</div>
        <ArrowRight className="lpe-arch-arrow" size={16} />
        <div className="lpe-arch-box input">Your tokens</div>
      </div>

      <p className="lpe-caption">
        The original 2017 paper ("Attention Is All You Need") introduced a full encoder–decoder pair for translation: an encoder
        reads the whole source sentence, and a decoder generates the output while also attending back to the encoder's output.
        Modern generative LLMs drop the encoder entirely and just stack decoder blocks, using <Term id="causalMasking">masked</Term> self-attention
        so a token can only look at itself and earlier tokens, never ahead. That masking is exactly what powers next-word prediction —
        toggle it live in the Attention step (the switch near the top of the page).
      </p>

      <div className="lpe-row-label" style={{ marginTop: 18 }}>Reference: the paper's base model hyperparameters</div>
      <div className="lpe-specs-grid">
        {PAPER_SPECS.map((s) => (
          <div key={s.k} className="lpe-spec-cell"><span>{s.k}</span><b>{s.v}</b></div>
        ))}
      </div>
    </div>
  );
}

/* ============================== STEP VIEWS ============================== */

function TokenizeView({ tokens }) {
  const [revealed, setRevealed] = useState(tokens.length);
  useEffect(() => {
    setRevealed(0);
    let i = 0;
    const t = setInterval(() => {
      i += 1;
      setRevealed(i);
      if (i >= tokens.length) clearInterval(t);
    }, 200);
    return () => clearInterval(t);
  }, [tokens.join('|')]);

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-row-label">Raw text</div>
      <div className="lpe-raw-text">{tokens.join(' ') || '—'}</div>
      <ArrowRight className="lpe-down-arrow" size={20} />
      <div className="lpe-row-label"><Term id="token">Tokens</Term> → <Term id="tokenId">IDs</Term></div>
      <div className="lpe-chip-row">
        {tokens.map((t, i) => (
          <div key={i} className={`lpe-token-reveal ${i < revealed ? 'in' : ''}`}>
            <TokenChip word={t} id={tokenId(t)} color={CATEGORY_COLOR[categoryOf(t)]} />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmbedView({ tokens }) {
  const [hoverIdx, setHoverIdx] = useState(null);
  const data = tokens.map((t, i) => ({ ...embedPosition(t), word: t, idx: i }));
  const grouped = {};
  data.forEach((d) => { grouped[d.category] = grouped[d.category] || []; grouped[d.category].push(d); });

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-row-label"><Term id="embedding">Meaning space</Term> — words with related meaning sit closer together</div>
      <div className="lpe-scatter-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#232B3E" />
            <XAxis type="number" dataKey="x" domain={[0, 100]} hide />
            <YAxis type="number" dataKey="y" domain={[0, 100]} hide />
            <ZAxis range={[220, 220]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{ background: '#161D2E', border: '1px solid #2A3349', borderRadius: 8, color: '#E8EAF0', fontFamily: 'IBM Plex Mono, monospace', fontSize: 12 }}
              formatter={(v, n, p) => [CATEGORY_NAME[p.payload.category], 'kind']}
              labelFormatter={() => ''}
            />
            {Object.entries(grouped).map(([cat, pts]) => (
              <Scatter key={cat} data={pts} fill={CATEGORY_COLOR[cat]}>
                {pts.map((p, i) => (
                  <Cell key={i} r={hoverIdx === p.idx ? 10 : 7} stroke={p.idx === hoverIdx ? '#fff' : 'none'} strokeWidth={2}
                    onMouseEnter={() => setHoverIdx(p.idx)} onMouseLeave={() => setHoverIdx(null)} />
                ))}
              </Scatter>
            ))}
          </ScatterChart>
        </ResponsiveContainer>
        <div className="lpe-scatter-labels">
          {data.map((d, i) => (
            <span key={i} className="lpe-scatter-label" style={{ left: `${d.x}%`, top: `${d.y}%`, color: CATEGORY_COLOR[d.category] }}
              onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}>{d.word}</span>
          ))}
        </div>
      </div>
      <div className="lpe-legend">
        {Object.entries(CATEGORY_COLOR).map(([cat, color]) => (
          <span key={cat} className="lpe-legend-item"><i style={{ background: color }} />{CATEGORY_NAME[cat]}</span>
        ))}
      </div>
      <p className="lpe-caption">These <Term id="embedding">meaning-vectors</Term> are what later get projected into Query, Key and Value vectors for the attention step.</p>
    </div>
  );
}

function PositionView({ tokens }) {
  const dims = 16;
  const chartData = Array.from({ length: dims }, (_, d) => {
    const row = { dim: d };
    tokens.forEach((t, i) => { row[`t${i}`] = Math.round(positionalEncodingSample(i, dims)[d] * 100) / 100; });
    return row;
  });

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-row-label">
        Each token's <Term id="embedding">meaning-vector</Term> gets a <Term id="positionalEncoding">positional encoding</Term> added to it
      </div>
      <div className="lpe-position-row">
        {tokens.map((t, i) => (
          <div key={i} className="lpe-position-card">
            <div className="lpe-position-badge">pos {i}</div>
            <div className="lpe-position-plus">+</div>
            <TokenChip word={t} color={CATEGORY_COLOR[categoryOf(t)]} />
          </div>
        ))}
      </div>

      <div className="lpe-info-card" style={{ marginTop: 20 }}>
        <div className="lpe-info-title">How to read the chart below</div>
        <p>
          Each colored line is one word from your sentence. The x-axis is one of the vector's dimensions (0–15 shown
          here, out of the real 512); the y-axis is that dimension's value, always between -1 and 1 because sine and
          cosine never go outside that range. The exact numbers don't matter — what matters is that <b>every position
          produces a differently-shaped line</b>, so the model can tell "1st word" apart from "5th word" purely from
          that shape, before it even reads what the word means.
        </p>
      </div>

      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 4 }}>
          <XAxis dataKey="dim" tick={{ fill: '#8992A9', fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis hide domain={[-1.2, 1.2]} />
          <Tooltip contentStyle={{ background: '#161D2E', border: '1px solid #2A3349', borderRadius: 8, fontSize: 11 }} />
          {tokens.map((t, i) => (
            <Line key={i} type="monotone" dataKey={`t${i}`} name={t} stroke={hslColor(i, tokens.length)} strokeWidth={2} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="lpe-legend">
        {tokens.map((t, i) => (
          <span key={i} className="lpe-legend-item"><i style={{ background: hslColor(i, tokens.length) }} />"{t}" (pos {i})</span>
        ))}
      </div>

      <p className="lpe-caption">
        The formula: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)). Sine and cosine
        were chosen because they're smooth, bounded, and repeat predictably — which the paper's authors found let the
        model generalize to sentence lengths longer than anything it saw during training.
      </p>
    </div>
  );
}

/* ---- Attention: Rays / Focus / Matrix / Heads ---- */

function RankedBars({ tokens, weights, selected }) {
  const max = Math.max(...weights.filter((_, i) => i !== selected), 0.0001);
  const ranked = tokens.map((t, i) => ({ t, w: weights[i], i })).filter((r) => r.i !== selected).sort((a, b) => b.w - a.w);
  return (
    <div className="lpe-bars lpe-bars-scroll">
      {ranked.map((r) => (
        <div key={r.i} className="lpe-bar-row">
          <span className="lpe-bar-label">{r.t}</span>
          <div className="lpe-bar-track">
            <div className="lpe-bar-fill" style={{ width: `${(r.w / max) * 100}%`, background: '#F2A93B' }} />
          </div>
          <span className="lpe-bar-val">{(r.w * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function AttentionRays({ tokens, selected, setSelected, causal }) {
  const weights = useMemo(() => computeAttention(tokens, selected, 1, causal), [tokens.join('|'), selected, causal]);
  const catColor = CATEGORY_COLOR[categoryOf(tokens[selected])];
  const ranked = tokens.map((t, i) => ({ i, w: weights[i] })).filter((r) => r.i !== selected).sort((a, b) => b.w - a.w).slice(0, Math.min(4, tokens.length - 1));
  const width = tokens.length * TOKEN_W;
  const H = 84;
  const baseY = H - 6;
  const xOf = (i) => i * TOKEN_W + TOKEN_W / 2;

  return (
    <div className="lpe-rays-scroll">
      <div className="lpe-rays-inner" style={{ width }}>
        <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} className="lpe-rays-svg">
          {ranked.map((r) => {
            const x1 = xOf(selected);
            const x2 = xOf(r.i);
            const peak = 14 + r.w * 55;
            const midX = (x1 + x2) / 2;
            const midY = baseY - peak;
            return (
              <g key={r.i}>
                <path d={`M ${x1} ${baseY} Q ${midX} ${midY} ${x2} ${baseY}`}
                  fill="none" stroke={catColor} strokeWidth={1.2 + r.w * 9} strokeOpacity={0.25 + r.w * 0.65} strokeLinecap="round" />
                <text x={midX} y={Math.max(10, midY - 4)} textAnchor="middle" fontSize="10" fill={catColor} fontFamily="IBM Plex Mono, monospace">
                  {(r.w * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}
        </svg>
        <div className="lpe-rays-tokens" style={{ width }}>
          {tokens.map((t, i) => (
            <div key={i} className="lpe-ray-slot" style={{ width: TOKEN_W }}>
              <button className={`lpe-token-btn ${i === selected ? 'active' : ''}`}
                style={{ borderColor: CATEGORY_COLOR[categoryOf(t)] }} onClick={() => setSelected(i)}>{t}</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AttentionMatrix({ tokens, onSelectRow, selected, causal }) {
  const matrix = useMemo(() => tokens.map((_, i) => computeAttention(tokens, i, 1, causal)), [tokens.join('|'), causal]);
  return (
    <div className="lpe-matrix-scroll">
      <table className="lpe-matrix-table">
        <thead>
          <tr>
            <th className="lpe-matrix-corner" />
            {tokens.map((t, j) => <th key={j} className="lpe-matrix-colhead">{t}</th>)}
          </tr>
        </thead>
        <tbody>
          {tokens.map((t, i) => (
            <tr key={i}>
              <th className={`lpe-matrix-rowhead ${i === selected ? 'active' : ''}`} onClick={() => onSelectRow(i)}>{t}</th>
              {tokens.map((_, j) => {
                const w = matrix[i][j];
                return (
                  <td key={j} className="lpe-matrix-cell" title={`"${t}" → "${tokens[j]}": ${(w * 100).toFixed(1)}%`}
                    style={{ background: `rgba(242,169,59,${Math.min(0.92, w * 2.2)})` }} />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HeadsView({ tokens, selected, setSelected, causal }) {
  return (
    <>
      <div className="lpe-row-label">4 simplified <Term id="multiHead">heads</Term> (the paper uses 8) — each specializes differently, then all of them are concatenated</div>
      <div className="lpe-chip-row lpe-attn-tokens">
        {tokens.map((t, i) => (
          <button key={i} className={`lpe-token-btn ${i === selected ? 'active' : ''}`}
            style={{ borderColor: CATEGORY_COLOR[categoryOf(t)] }} onClick={() => setSelected(i)}>{t}</button>
        ))}
      </div>
      <div className="lpe-heads-grid">
        {HEAD_META.map((meta, h) => {
          const w = computeAttentionHead(tokens, selected, h, causal);
          const top = tokens.map((t, i) => ({ t, w: w[i], i })).filter((r) => r.i !== selected).sort((a, b) => b.w - a.w)[0];
          return (
            <div key={h} className="lpe-head-card" style={{ borderColor: `${meta.color}55` }}>
              <div className="lpe-head-title" style={{ color: meta.color }}>{meta.label}</div>
              <div className="lpe-head-role">{meta.role}</div>
              <div className="lpe-chip-row" style={{ marginTop: 8 }}>
                {tokens.map((t, i) => {
                  const isSel = i === selected;
                  const ww = isSel ? 1 : w[i];
                  const alpha = Math.round(Math.min(0.85, ww * 1.7) * 255).toString(16).padStart(2, '0');
                  return (
                    <span key={i} className="lpe-head-tok" style={{
                      background: isSel ? 'transparent' : `${meta.color}${alpha}`,
                      borderColor: isSel ? meta.color : 'transparent',
                      color: isSel ? meta.color : (ww > 0.32 ? '#0d1220' : '#cfd3e0')
                    }}>{t}</span>
                  );
                })}
              </div>
              <div className="lpe-head-top">Top: <b>{top.t}</b> ({(top.w * 100).toFixed(0)}%)</div>
            </div>
          );
        })}
      </div>
      <p className="lpe-caption">These 4 (of 8 in the paper) run in parallel on the same input, then get concatenated and projected back down — letting the model track several relationship types at once, something a single attention pass can't do.</p>
    </>
  );
}

function AttentionView({ tokens, selected, setSelected, causal }) {
  const [mode, setMode] = useState('rays');
  if (tokens.length < 2) {
    return <div className="lpe-stage-inner"><EmptyState>Add at least one more word to see how tokens attend to each other.</EmptyState></div>;
  }
  const weights = useMemo(() => computeAttention(tokens, selected, 1, causal), [tokens.join('|'), selected, causal]);

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-qkv-card">
        <div className="lpe-qkv-formula">Attention(Q, K, V) = <Term id="softmax">softmax</Term>( Q·Kᵀ / √d_k ) · V</div>
        <div className="lpe-qkv-grid">
          <div><Term id="query"><b>Query (Q)</b></Term> — what this token is looking for</div>
          <div><Term id="key"><b>Key (K)</b></Term> — what each token offers, to be matched against</div>
          <div><Term id="value"><b>Value (V)</b></Term> — the content actually passed along once matched</div>
        </div>
        <p className="lpe-info-note">
          Toggle above: <Term id="causalMasking">Causal</Term> restricts attention to earlier words only (how GPT works);
          Full lets tokens see the whole sentence (how BERT works).
        </p>
      </div>

      <div className="lpe-tab-row">
        <button className={`lpe-tab ${mode === 'rays' ? 'active' : ''}`} onClick={() => setMode('rays')}>Rays view</button>
        <button className={`lpe-tab ${mode === 'focus' ? 'active' : ''}`} onClick={() => setMode('focus')}>Focus view</button>
        <button className={`lpe-tab ${mode === 'matrix' ? 'active' : ''}`} onClick={() => setMode('matrix')}>Matrix view</button>
        <button className={`lpe-tab ${mode === 'heads' ? 'active' : ''}`} onClick={() => setMode('heads')}>Heads view</button>
      </div>

      {mode === 'rays' && (
        <>
          <div className="lpe-row-label">Click a token — curved rays trace its strongest connections (scroll for long sentences)</div>
          <AttentionRays tokens={tokens} selected={selected} setSelected={setSelected} causal={causal} />
          <div className="lpe-row-label" style={{ marginTop: 20 }}>Ranked weights for "<b>{tokens[selected]}</b>"</div>
          <RankedBars tokens={tokens} weights={weights} selected={selected} />
        </>
      )}

      {mode === 'focus' && (
        <>
          <div className="lpe-row-label">Click a token — brighter highlight on the others = stronger attention from it</div>
          <div className="lpe-highlight-wrap">
            {tokens.map((t, i) => {
              const isSel = i === selected;
              const w = isSel ? 1 : weights[i];
              return (
                <button key={i} className={`lpe-hl-token ${isSel ? 'selected' : ''}`}
                  style={!isSel ? { background: `rgba(242,169,59,${Math.min(0.85, w * 1.7)})`, color: w > 0.32 ? '#16130a' : '#cfd3e0' } : undefined}
                  onClick={() => setSelected(i)}>{t}</button>
              );
            })}
          </div>
          <div className="lpe-row-label" style={{ marginTop: 20 }}>Ranked weights for "<b>{tokens[selected]}</b>"</div>
          <RankedBars tokens={tokens} weights={weights} selected={selected} />
        </>
      )}

      {mode === 'matrix' && (
        <>
          <div className="lpe-row-label">Every token's attention to every other token — darker cell = stronger weight (simulated)</div>
          <AttentionMatrix tokens={tokens} selected={selected} onSelectRow={setSelected} causal={causal} />
        </>
      )}

      {mode === 'heads' && <HeadsView tokens={tokens} selected={selected} setSelected={setSelected} causal={causal} />}
    </div>
  );
}

/* ---- Layers ---- */

function LayersView({ tokens, selected, setSelected, causal }) {
  if (tokens.length < 2) {
    return <div className="lpe-stage-inner"><EmptyState>Add at least one more word to explore layer-by-layer focus.</EmptyState></div>;
  }
  const N = N_LAYERS;
  const [layer, setLayer] = useState(1);

  const layerData = useMemo(() => {
    return Array.from({ length: N }, (_, idx) => {
      const l = idx + 1;
      const w = computeAttention(tokens, selected, 1 + l * 0.6, causal);
      const ranked = tokens.map((t, i) => ({ t, w: w[i], i })).filter((r) => r.i !== selected).sort((a, b) => b.w - a.w);
      return { layer: l, top: ranked[0], top3: ranked.slice(0, 3), confidence: ranked[0].w };
    });
  }, [tokens.join('|'), selected, causal]);

  const current = layerData[layer - 1];
  const prev = layer > 1 ? layerData[layer - 2] : null;

  let changeMsg;
  if (!prev) {
    changeMsg = 'First pass — focus is still fairly broad across nearby words.';
  } else {
    const deltaPts = Math.round((current.confidence - prev.confidence) * 100);
    if (current.top.t !== prev.top.t) {
      changeMsg = `Focus shifted from "${prev.top.t}" to "${current.top.t}" — confidence ${deltaPts >= 0 ? 'rose' : 'fell'} by ${Math.abs(deltaPts)} pts.`;
    } else {
      changeMsg = `Kept focus on "${current.top.t}", but confidence ${deltaPts >= 0 ? 'sharpened by' : 'eased by'} ${Math.abs(deltaPts)} pts.`;
    }
  }

  const trendData = layerData.map((d) => ({ name: `L${d.layer}`, confidence: Math.round(d.confidence * 1000) / 10 }));

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-block-anatomy">
        <span className="lpe-anatomy-box attn"><Term id="causalMasking">Masked</Term> <Term id="multiHead">Multi-Head Attention</Term></span>
        <ArrowRight size={12} className="lpe-anatomy-arrow" />
        <span className="lpe-anatomy-box norm"><Term id="addNorm">Add & Norm</Term></span>
        <ArrowRight size={12} className="lpe-anatomy-arrow" />
        <span className="lpe-anatomy-box ff"><Term id="feedForward">Feed-Forward (ReLU)</Term></span>
        <ArrowRight size={12} className="lpe-anatomy-arrow" />
        <span className="lpe-anatomy-box norm"><Term id="addNorm">Add & Norm</Term></span>
      </div>
      <p className="lpe-caption" style={{ marginTop: 4, marginBottom: 16 }}>
        Every layer repeats this same anatomy, stacked {N_LAYERS} times end to end. Tap any of the labels above for a
        plain-language definition — together they're the entire "engine" that gets reused over and over.
      </p>

      <div className="lpe-row-label">Focus token</div>
      <div className="lpe-chip-row lpe-attn-tokens">
        {tokens.map((t, i) => (
          <button key={i} className={`lpe-token-btn ${i === selected ? 'active' : ''}`}
            style={{ borderColor: CATEGORY_COLOR[categoryOf(t)] }} onClick={() => setSelected(i)}>{t}</button>
        ))}
      </div>

      <div className="lpe-layers-body">
        <div className="lpe-layer-stack">
          {Array.from({ length: N }).map((_, i) => {
            const l = N - i;
            return (
              <div key={l} className={`lpe-layer-card ${l === layer ? 'active' : ''}`} onClick={() => setLayer(l)}>
                <span>Layer {l}</span>
                {l === layer && <span className="lpe-layer-dot" />}
              </div>
            );
          })}
        </div>

        <div className="lpe-layer-detail">
          <input type="range" min={1} max={N} value={layer} onChange={(e) => setLayer(Number(e.target.value))} className="lpe-slider" />

          <div className="lpe-layer-role"><Sparkles size={14} /> <span>{layerRoleText(layer, N)}</span></div>

          <div className="lpe-row-label" style={{ marginBottom: 2 }}><Term id="confidence">Confidence</Term> trend across layers</div>
          <div className="lpe-layer-trend">
            <ResponsiveContainer width="100%" height={90}>
              <LineChart data={trendData} margin={{ top: 6, right: 14, left: 0, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fill: '#8992A9', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis hide domain={[0, 'dataMax + 10']} />
                <Tooltip contentStyle={{ background: '#161D2E', border: '1px solid #2A3349', borderRadius: 8, color: '#E8EAF0', fontSize: 12 }} formatter={(v) => [`${v}%`, 'confidence']} />
                <Line type="monotone" dataKey="confidence" stroke="#F2A93B" strokeWidth={2} dot={{ r: 4, fill: '#F2A93B' }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="lpe-change-note"><ArrowRight size={14} /> <span>{changeMsg}</span></div>

          <div className="lpe-row-label" style={{ marginTop: 14 }}>Top tokens at Layer {layer}</div>
          <div className="lpe-chip-row">
            {current.top3.map((s, i) => {
              let trend = null;
              if (prev) {
                const prevRank = prev.top3.findIndex((p) => p.i === s.i);
                if (prevRank === -1) trend = 'new';
                else if (prevRank > i) trend = 'up';
                else if (prevRank < i) trend = 'down';
                else trend = 'same';
              }
              return (
                <div key={i} className="lpe-rank-chip">
                  <span className="lpe-rank-num">#{i + 1}</span>
                  <TokenChip word={s.t} color={CATEGORY_COLOR[categoryOf(s.t)]} />
                  <span className="lpe-bar-val">{(s.w * 100).toFixed(0)}%</span>
                  {trend && <span className={`lpe-trend-tag ${trend}`}>{trend === 'up' ? '▲' : trend === 'down' ? '▼' : trend === 'new' ? 'NEW' : '–'}</span>}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---- Output ---- */

function OutputView({ tokens, sentence, onGenerate }) {
  const preds = useMemo(() => predictNext(tokens), [tokens.join('|')]);
  const max = preds[0][1];
  const data = preds.map(([word, p]) => ({ word, p: Math.round(p * 1000) / 10 }));

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-info-card">
        <div className="lpe-info-title">How to read this step</div>
        <p>
          The model has finished processing your sentence and now scores every word in its <Term id="vocabulary">vocabulary</Term> —
          real models score 50,000+ candidates at once. A function called <Term id="softmax">softmax</Term> turns those raw
          scores into <Term id="probability">probabilities</Term> that add up to 100%. The bar chart below shows the top
          candidates, longest bar first — that's the model's single most likely next word.
        </p>
      </div>

      <div className="lpe-row-label" style={{ marginTop: 16 }}>Prompt: <span className="lpe-mono">"{sentence} ___"</span></div>
      <div className="lpe-output-chart">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30, top: 5, bottom: 5 }}>
            <XAxis type="number" hide domain={[0, Math.max(40, max * 100 + 5)]} />
            <YAxis type="category" dataKey="word" width={90}
              tick={{ fill: '#E8EAF0', fontFamily: 'IBM Plex Mono, monospace', fontSize: 13 }} axisLine={false} tickLine={false} />
            <Tooltip
              cursor={{ fill: '#ffffff08' }}
              contentStyle={{ background: '#161D2E', border: '1px solid #2A3349', borderRadius: 8, color: '#E8EAF0' }}
              formatter={(v) => [`${v}%`, 'probability']} />
            <Bar dataKey="p" radius={[0, 6, 6, 0]}>
              {data.map((d, i) => <Cell key={i} fill={i === 0 ? '#F2A93B' : '#3A4560'} />)}
              <LabelList dataKey="p" position="right" formatter={(v) => `${v}%`} fill="#8992A9" fontSize={12} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="lpe-output-actions">
        <button className="lpe-generate-btn" onClick={() => onGenerate(preds[0][0])}>
          <Wand2 size={16} /> Generate "{preds[0][0]}" and continue
        </button>
        <span className="lpe-pill"><Shield size={12} /> <Term id="repetitionPenalty">Repetition penalty</Term>: active</span>
      </div>
      <p className="lpe-caption">
        Clicking "Generate" appends the top word to your sentence and re-runs the entire pipeline from the top — this loop
        (called autoregressive generation) is exactly how tools like ChatGPT write whole paragraphs, one word at a time.
        This demo simplifies the real vocabulary and scoring down to a handful of plausible words for clarity.
      </p>
    </div>
  );
}

/* ---- Full Pipeline (checkpoint-by-checkpoint walkthrough) ---- */

function checkpointContent(id, tokens, sentence, selected, causal) {
  switch (id) {
    case 'input':
      return { caption: 'The raw text you typed, before any processing.', body: <div className="lpe-raw-text" style={{ fontSize: 13 }}>{sentence}</div> };
    case 'tokenize':
      return {
        caption: `Split into ${tokens.length} token${tokens.length === 1 ? '' : 's'} and mapped to IDs.`,
        body: <div className="lpe-chip-row">{tokens.map((t, i) => <TokenChip key={i} word={t} id={tokenId(t)} color={CATEGORY_COLOR[categoryOf(t)]} />)}</div>
      };
    case 'embed': {
      const cats = [...new Set(tokens.map(categoryOf))];
      return {
        caption: `Each token became a meaning-vector — ${cats.length} kind${cats.length === 1 ? '' : 's'} of word detected.`,
        body: <div className="lpe-chip-row">{tokens.map((t, i) => <TokenChip key={i} word={t} color={CATEGORY_COLOR[categoryOf(t)]} />)}</div>
      };
    }
    case 'position':
      return {
        caption: "Order information (sin/cos waves) added so word sequence isn't lost.",
        body: (
          <div className="lpe-chip-row">
            {tokens.map((t, i) => (
              <div key={i} className="lpe-mini-pos"><span>{i}</span><TokenChip word={t} color={CATEGORY_COLOR[categoryOf(t)]} /></div>
            ))}
          </div>
        )
      };
    case 'attention': {
      if (tokens.length < 2) return { caption: 'Needs at least 2 tokens to compare.', body: null };
      const idx = Math.min(selected, tokens.length - 1);
      const w = computeAttention(tokens, idx, 1, causal);
      const top = tokens.map((t, i) => ({ t, w: w[i], i })).filter((r) => r.i !== idx).sort((a, b) => b.w - a.w)[0];
      return {
        caption: `"${tokens[idx]}" pays the most attention to "${top.t}" (${(top.w * 100).toFixed(0)}%)${causal ? ' — masked, so only earlier words are visible' : ''}.`,
        body: <div className="lpe-chip-row">{tokens.map((t, i) => <TokenChip key={i} word={t} color={CATEGORY_COLOR[categoryOf(t)]} faded={i !== idx && i !== top.i} />)}</div>
      };
    }
    case 'layers': {
      if (tokens.length < 2) return { caption: 'Needs at least 2 tokens to compare.', body: null };
      const idx = Math.min(selected, tokens.length - 1);
      const c1 = Math.max(...computeAttention(tokens, idx, 1.6, causal));
      const cN = Math.max(...computeAttention(tokens, idx, 1 + N_LAYERS * 0.6, causal));
      return {
        caption: `Confidence sharpened from ${(c1 * 100).toFixed(0)}% at Layer 1 to ${(cN * 100).toFixed(0)}% by Layer ${N_LAYERS}.`,
        body: <div className="lpe-mini-trend"><span>L1 · {(c1 * 100).toFixed(0)}%</span><ArrowRight size={12} /><span>L{N_LAYERS} · {(cN * 100).toFixed(0)}%</span></div>
      };
    }
    case 'output': {
      const preds = predictNext(tokens);
      return {
        caption: `Top prediction: "${preds[0][0]}" at ${(preds[0][1] * 100).toFixed(0)}% probability.`,
        body: <span className="lpe-pill lpe-pill-solid">{preds[0][0]} · {(preds[0][1] * 100).toFixed(0)}%</span>
      };
    }
    default:
      return { caption: '', body: null };
  }
}

function PipelineView({ tokens, sentence, selected, playing, speed, causal, onJump, onFinish }) {
  const [reveal, setReveal] = useState(0);
  useEffect(() => { setReveal(0); }, [sentence]);

  useEffect(() => {
    if (!playing) return undefined;
    if (reveal >= PIPELINE_NODES.length - 1) { onFinish && onFinish(); return undefined; }
    const t = setTimeout(() => setReveal((r) => Math.min(PIPELINE_NODES.length - 1, r + 1)), speed);
    return () => clearTimeout(t);
  }, [playing, reveal, speed]);

  return (
    <div className="lpe-stage-inner">
      <div className="lpe-row-label">Watch your sentence move through every stage, checkpoint by checkpoint</div>
      <div className="lpe-tl-controls">
        <button className="lpe-icon-btn" onClick={() => setReveal(0)}><RotateCcw size={13} /> Restart</button>
        <button className="lpe-icon-btn" onClick={() => setReveal((r) => Math.max(0, r - 1))}>◀ Back</button>
        <button className="lpe-icon-btn" onClick={() => setReveal((r) => Math.min(PIPELINE_NODES.length - 1, r + 1))}>Next ▶</button>
        <span className="lpe-pill">Checkpoint {reveal} / {PIPELINE_NODES.length - 1}</span>
      </div>

      <div className="lpe-tl">
        {PIPELINE_NODES.map((node, i) => {
          const done = i <= reveal;
          const isCurrent = i === reveal;
          const content = done ? checkpointContent(node.id, tokens, sentence, selected, causal) : null;
          return (
            <div key={node.id} className={`lpe-tl-item ${done ? 'done' : ''} ${isCurrent ? 'current' : ''}`}>
              <div className="lpe-tl-rail">
                <div className="lpe-tl-dot">{i}</div>
                {i < PIPELINE_NODES.length - 1 && <div className="lpe-tl-line" />}
              </div>
              <div className={`lpe-tl-body ${node.id !== 'input' ? 'clickable' : ''}`} onClick={() => node.id !== 'input' && onJump(node.id)}>
                <div className="lpe-tl-title">{node.label}</div>
                {done ? (
                  <>
                    <div className="lpe-tl-caption">{content.caption}</div>
                    {content.body}
                  </>
                ) : (
                  <div className="lpe-tl-caption lpe-tl-waiting">Waiting…</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============================== APP ============================== */

export default function App() {
  const [sentence, setSentence] = useState(DEFAULT_SENTENCE);
  const [draft, setDraft] = useState(DEFAULT_SENTENCE);
  const [activeIdx, setActiveIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1600);
  const [selected, setSelected] = useState(0);
  const [causal, setCausal] = useState(true);
  const [openTermId, setOpenTermId] = useState(null);
  const timerRef = useRef(null);

  const tokens = useMemo(() => tokenize(sentence), [sentence]);
  const step = STEPS[activeIdx];

  useEffect(() => { setSelected(Math.max(0, tokens.length - 1)); }, [sentence]);

  useEffect(() => {
    if (!playing) { clearInterval(timerRef.current); return undefined; }
    if (step.id === 'pipeline') { clearInterval(timerRef.current); return undefined; } // handled internally by PipelineView
    timerRef.current = setInterval(() => {
      setActiveIdx((i) => {
        if (i >= STEPS.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, speed);
    return () => clearInterval(timerRef.current);
  }, [playing, speed, step.id]);

  function commitSentence(text) {
    const clean = (text !== undefined ? text : draft).trim() || DEFAULT_SENTENCE;
    setSentence(clean);
    setDraft(clean);
    setActiveIdx(0);
    setPlaying(false);
  }

  function handleGenerate(word) {
    const next = `${sentence} ${word}`;
    setSentence(next);
    setDraft(next);
    setActiveIdx(0);
    setPlaying(true);
  }

  function jumpTo(id) {
    const idx = STEPS.findIndex((s) => s.id === id);
    if (idx >= 0) { setActiveIdx(idx); setPlaying(false); }
  }

  let extraNote = '';
  if (step.id === 'attention' && tokens.length > 1) extraNote = `Try clicking "${tokens[Math.min(1, tokens.length - 1)] || ''}" vs "${tokens[tokens.length - 1] || ''}" and compare which words light up.`;
  if (step.id === 'output') extraNote = 'Hit generate to feed the predicted word back in — this loop is exactly how models write full sentences.';
  if (step.id === 'pipeline') extraNote = 'Hit the Play button up top to watch every checkpoint reveal itself in sequence, end to end.';
  if (step.id === 'architecture') extraNote = 'Head to the Attention step to toggle between causal (GPT-style) and full (BERT-style) masking live.';

  return (
    <TermContext.Provider value={{ openId: openTermId, setOpenId: setOpenTermId }}>
    <div className="lpe-root" onClick={() => setOpenTermId(null)}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

        .lpe-root {
          --bg: #0F1420; --panel: #161D2E; --border: #232B3E; --border2: #2A3349;
          --text: #E8EAF0; --muted: #8992A9; --amber: #F2A93B; --teal: #5EEAD4;
          font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text);
          min-height: 100%; padding: 20px; box-sizing: border-box;
        }
        .lpe-root * { box-sizing: border-box; }
        .lpe-header { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; margin-bottom: 10px; }
        .lpe-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 20px; margin: 0; letter-spacing: -0.01em; display: flex; align-items: center; gap: 8px; }
        .lpe-title span { color: var(--amber); }
        .lpe-logo-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--amber); box-shadow: 0 0 10px var(--amber); flex-shrink: 0; }
        .lpe-subtitle { color: var(--muted); font-size: 13px; margin: 2px 0 0; }
        .lpe-badge {
          font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--teal); border: 1px solid #2A3349;
          border-radius: 20px; padding: 3px 10px; white-space: nowrap;
        }
        .lpe-input-row { display: flex; gap: 8px; align-items: center; flex: 1; min-width: 260px; }
        .lpe-input {
          flex: 1; background: var(--panel); border: 1px solid var(--border2); color: var(--text);
          border-radius: 8px; padding: 9px 12px; font-family: 'IBM Plex Mono', monospace; font-size: 13px;
        }
        .lpe-input:focus { outline: none; border-color: var(--amber); }
        .lpe-icon-btn {
          background: var(--panel); border: 1px solid var(--border2); color: var(--text);
          border-radius: 8px; padding: 8px 10px; cursor: pointer; display: flex; align-items: center; gap: 6px;
          font-size: 13px; font-family: 'Inter', sans-serif; transition: border-color .15s, background .15s;
        }
        .lpe-icon-btn:hover { border-color: var(--amber); background: #1c2436; }
        .lpe-icon-btn.primary { background: var(--amber); color: #16130a; border-color: var(--amber); font-weight: 600; }
        .lpe-icon-btn.primary:hover { background: #ffbd5c; }
        .lpe-select {
          background: var(--panel); border: 1px solid var(--border2); color: var(--text);
          border-radius: 8px; padding: 8px 10px; font-size: 13px;
        }
        .lpe-mask-toggle { display: flex; border: 1px solid var(--border2); border-radius: 8px; overflow: hidden; }
        .lpe-mask-btn {
          background: var(--panel); border: none; color: var(--muted); padding: 8px 12px; font-size: 12px;
          cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background .15s, color .15s;
        }
        .lpe-mask-btn.active { background: var(--amber); color: #16130a; font-weight: 600; }
        .lpe-presets { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 16px; }
        .lpe-preset-chip {
          background: transparent; border: 1px dashed var(--border2); color: var(--muted); border-radius: 20px;
          padding: 5px 12px; font-size: 12px; cursor: pointer; transition: border-color .15s, color .15s;
        }
        .lpe-preset-chip:hover { border-color: var(--amber); color: var(--amber); }

        .lpe-body { display: grid; grid-template-columns: 168px 1fr; gap: 16px; }
        @media (max-width: 760px) { .lpe-body { grid-template-columns: 1fr; } }

        .lpe-rail { display: flex; flex-direction: column; gap: 6px; }
        .lpe-rail-item {
          display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 8px;
          border: 1px solid transparent; cursor: pointer; background: transparent; color: var(--muted);
          font-size: 13px; text-align: left; transition: background .15s, color .15s, border-color .15s;
        }
        .lpe-rail-item:hover { background: #161d2e; color: var(--text); }
        .lpe-rail-item.active { background: var(--panel); border-color: var(--border2); color: var(--text); }
        .lpe-rail-num {
          width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
          font-family: 'IBM Plex Mono', monospace; font-size: 11px; border: 1px solid var(--border2); flex-shrink: 0;
        }
        .lpe-rail-item.active .lpe-rail-num { background: var(--amber); color: #16130a; border-color: var(--amber); }

        .lpe-main { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
        .lpe-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .lpe-stage-title { font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 600; margin: 0 0 14px; }
        .lpe-stage-inner { min-height: 260px; }
        .lpe-empty { color: var(--muted); font-size: 13.5px; padding: 40px 10px; text-align: center; }

        .lpe-row-label { color: var(--muted); font-size: 12.5px; margin-bottom: 10px; }
        .lpe-raw-text { font-family: 'IBM Plex Mono', monospace; font-size: 15px; background: #0d1220; border: 1px solid var(--border2); border-radius: 8px; padding: 10px 14px; display: inline-block; }
        .lpe-down-arrow { color: var(--muted); margin: 10px 0; transform: rotate(90deg); }
        .lpe-chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .lpe-token-reveal { opacity: 0; transform: translateY(6px); transition: opacity .3s, transform .3s; }
        .lpe-token-reveal.in { opacity: 1; transform: translateY(0); }

        .lpe-chip {
          display: flex; flex-direction: column; align-items: center; gap: 2px;
          border: 1.5px solid; border-radius: 8px; padding: 6px 12px; background: #0d1220; min-width: 46px;
        }
        .lpe-chip-word { font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
        .lpe-chip-id { font-size: 10px; opacity: .85; }

        .lpe-scatter-wrap { position: relative; }
        .lpe-scatter-labels { position: absolute; inset: 0; pointer-events: none; }
        .lpe-scatter-label {
          position: absolute; transform: translate(-50%, 12px); font-family: 'IBM Plex Mono', monospace;
          font-size: 11px; pointer-events: auto; cursor: default; white-space: nowrap;
        }
        .lpe-legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 6px; }
        .lpe-legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
        .lpe-legend-item i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

        .lpe-position-row { display: flex; flex-wrap: wrap; gap: 14px; }
        .lpe-position-card { display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .lpe-position-badge { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--teal); border: 1px solid #2A3349; border-radius: 6px; padding: 2px 6px; }
        .lpe-position-plus { color: var(--muted); font-size: 12px; }
        .lpe-caption { color: var(--muted); font-size: 12.5px; margin-top: 14px; line-height: 1.5; }

        .lpe-qkv-card { background: #0d1220; border: 1px solid var(--border2); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; }
        .lpe-qkv-formula { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--amber); margin-bottom: 8px; }
        .lpe-qkv-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 11.5px; color: #cfd3e0; line-height: 1.4; }
        @media (max-width: 700px) { .lpe-qkv-grid { grid-template-columns: 1fr; } }
        .lpe-info-note { margin: 10px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.5; }

        .lpe-info-card { background: #0d1220; border: 1px solid var(--border2); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; }
        .lpe-info-title { font-family: 'Space Grotesk', sans-serif; font-size: 12.5px; font-weight: 600; color: var(--amber); margin-bottom: 6px; }
        .lpe-info-card p { margin: 0; font-size: 12.5px; line-height: 1.6; color: #cfd3e0; }

        .lpe-term-wrap { position: relative; display: inline-block; }
        .lpe-term-btn { background: none; border: none; padding: 0; margin: 0; font: inherit; color: inherit; cursor: pointer; }
        .lpe-term-i { font-size: 0.72em; color: var(--amber); margin-left: 2px; vertical-align: super; }
        .lpe-term-pop {
          position: absolute; z-index: 50; left: 0; top: 100%; margin-top: 6px; width: 230px; max-width: 72vw;
          background: #161D2E; border: 1px solid #2A3349; border-radius: 8px; padding: 10px 12px;
          font-family: 'Inter', sans-serif; font-weight: 400; font-size: 12px; line-height: 1.5; color: #E8EAF0;
          box-shadow: 0 10px 26px rgba(0,0,0,.45); text-align: left; white-space: normal;
        }

        .lpe-tab-row { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
        .lpe-tab {
          background: transparent; border: 1px solid var(--border2); color: var(--muted); border-radius: 8px;
          padding: 6px 14px; font-size: 12.5px; cursor: pointer; transition: all .15s;
        }
        .lpe-tab.active { background: var(--amber); color: #16130a; border-color: var(--amber); font-weight: 600; }

        .lpe-rays-scroll { overflow-x: auto; display: flex; justify-content: center; border: 1px solid var(--border2); border-radius: 10px; background: #0d1220; padding: 10px 6px 14px; }
        .lpe-rays-inner { position: relative; flex-shrink: 0; }
        .lpe-rays-svg { display: block; }
        .lpe-rays-tokens { display: flex; }
        .lpe-ray-slot { display: flex; justify-content: center; flex-shrink: 0; }

        .lpe-highlight-wrap { display: flex; flex-wrap: wrap; gap: 8px; }
        .lpe-hl-token {
          font-family: 'IBM Plex Mono', monospace; font-size: 13px; border: 1.5px solid var(--border2); border-radius: 8px;
          padding: 7px 13px; cursor: pointer; background: #0d1220; color: var(--text); transition: background .25s, color .25s, transform .12s;
        }
        .lpe-hl-token:hover { transform: translateY(-1px); }
        .lpe-hl-token.selected { border-color: var(--amber); color: var(--amber); box-shadow: 0 0 0 2px #f2a93b22 inset; font-weight: 600; }

        .lpe-attn-tokens { margin-bottom: 16px; }
        .lpe-token-btn {
          font-family: 'IBM Plex Mono', monospace; font-size: 13px; background: #0d1220; color: var(--text);
          border: 1.5px solid var(--border2); border-radius: 8px; padding: 6px 12px; cursor: pointer; transition: transform .12s;
        }
        .lpe-token-btn:hover { transform: translateY(-1px); }
        .lpe-token-btn.active { background: #fff2; box-shadow: 0 0 0 2px #fff3 inset; }

        .lpe-bars { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
        .lpe-bars-scroll { max-height: 240px; overflow-y: auto; padding-right: 6px; }
        .lpe-bar-row { display: grid; grid-template-columns: 76px 1fr 46px; align-items: center; gap: 10px; }
        .lpe-bar-label { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .lpe-bar-track { background: #0d1220; border: 1px solid var(--border2); border-radius: 6px; height: 12px; overflow: hidden; }
        .lpe-bar-fill { height: 100%; border-radius: 6px; transition: width .35s ease; }
        .lpe-bar-val { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted); text-align: right; }

        .lpe-matrix-scroll { max-height: 380px; overflow: auto; border: 1px solid var(--border2); border-radius: 10px; background: #0d1220; }
        .lpe-matrix-table { border-collapse: collapse; }
        .lpe-matrix-table th, .lpe-matrix-table td { padding: 0; }
        .lpe-matrix-corner { position: sticky; top: 0; left: 0; z-index: 3; background: #0d1220; min-width: 76px; }
        .lpe-matrix-colhead {
          position: sticky; top: 0; z-index: 2; background: #0d1220; font-family: 'IBM Plex Mono', monospace;
          font-size: 11px; color: var(--muted); padding: 8px 6px !important; white-space: nowrap; border-bottom: 1px solid var(--border2);
        }
        .lpe-matrix-rowhead {
          position: sticky; left: 0; z-index: 1; background: #0d1220; font-family: 'IBM Plex Mono', monospace;
          font-size: 11px; color: var(--muted); padding: 6px 10px !important; text-align: right; white-space: nowrap;
          border-right: 1px solid var(--border2); cursor: pointer;
        }
        .lpe-matrix-rowhead.active { color: var(--amber); font-weight: 600; }
        .lpe-matrix-cell { width: 32px; height: 26px; border: 1px solid #0d1220; }

        .lpe-heads-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 6px; }
        @media (max-width: 600px) { .lpe-heads-grid { grid-template-columns: 1fr; } }
        .lpe-head-card { border: 1px solid; border-radius: 10px; padding: 12px; background: #0d1220; }
        .lpe-head-title { font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 600; }
        .lpe-head-role { color: var(--muted); font-size: 11px; margin-top: 2px; }
        .lpe-head-tok { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; border: 1.5px solid; border-radius: 6px; padding: 3px 7px; }
        .lpe-head-top { margin-top: 10px; font-size: 11.5px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }

        .lpe-layers-body { display: grid; grid-template-columns: 140px 1fr; gap: 18px; margin-top: 10px; }
        @media (max-width: 600px) { .lpe-layers-body { grid-template-columns: 1fr; } }
        .lpe-layer-stack { display: flex; flex-direction: column-reverse; gap: 6px; max-height: 320px; overflow-y: auto; }
        .lpe-layer-card {
          background: #0d1220; border: 1.5px solid var(--border2); border-radius: 8px; padding: 10px 12px;
          font-family: 'IBM Plex Mono', monospace; font-size: 12px; cursor: pointer; display: flex; justify-content: space-between; align-items: center;
        }
        .lpe-layer-card.active { border-color: var(--amber); background: #24200f; }
        .lpe-layer-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); }
        .lpe-slider { width: 100%; accent-color: var(--amber); }
        .lpe-layer-role {
          display: flex; gap: 8px; align-items: flex-start; color: var(--teal); font-size: 12.5px;
          background: #0d1220; border: 1px solid var(--border2); border-radius: 8px; padding: 10px 12px; margin: 12px 0; line-height: 1.5;
        }
        .lpe-layer-trend { margin: 2px 0 8px; }
        .lpe-change-note { display: flex; gap: 8px; align-items: center; color: var(--amber); font-size: 12.5px; margin: 4px 0 4px; }
        .lpe-rank-chip { display: flex; align-items: center; gap: 8px; background: #0d1220; border: 1px solid var(--border2); border-radius: 8px; padding: 6px 10px; }
        .lpe-rank-num { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted); }
        .lpe-trend-tag { font-size: 10px; padding: 1px 5px; border-radius: 4px; margin-left: 2px; font-family: 'IBM Plex Mono', monospace; }
        .lpe-trend-tag.up { color: #4ADE80; }
        .lpe-trend-tag.down { color: #FB7185; }
        .lpe-trend-tag.new { color: #F2A93B; }
        .lpe-trend-tag.same { color: var(--muted); }

        .lpe-block-anatomy { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 4px; }
        .lpe-anatomy-box { font-family: 'IBM Plex Mono', monospace; font-size: 11px; border: 1.5px solid var(--border2); border-radius: 6px; padding: 5px 9px; background: #0d1220; }
        .lpe-anatomy-box.attn { border-color: #F2A93B; color: #F2A93B; }
        .lpe-anatomy-box.norm { border-color: #5EEAD4; color: #5EEAD4; }
        .lpe-anatomy-box.ff { border-color: #FB7185; color: #FB7185; }
        .lpe-anatomy-arrow { color: var(--muted); flex-shrink: 0; }

        .lpe-arch-stack { display: flex; flex-direction: column; align-items: center; gap: 4px; max-width: 440px; margin: 0 auto 18px; }
        .lpe-arch-box { width: 100%; text-align: center; padding: 9px 12px; border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; border: 1.5px solid var(--border2); background: #0d1220; }
        .lpe-arch-box.attn { border-color: #F2A93B; color: #F2A93B; }
        .lpe-arch-box.norm { border-color: #5EEAD4; color: #5EEAD4; font-size: 11.5px; }
        .lpe-arch-box.ff { border-color: #FB7185; color: #FB7185; }
        .lpe-arch-box.output, .lpe-arch-box.input { color: var(--text); font-weight: 600; }
        .lpe-arch-arrow { color: var(--muted); }
        .lpe-arch-repeat { border: 1.5px dashed var(--border2); border-radius: 10px; padding: 10px; display: flex; flex-direction: column; gap: 6px; position: relative; width: 100%; margin: 2px 0; }
        .lpe-arch-repeat-label { position: absolute; right: 10px; bottom: -10px; background: var(--panel); padding: 0 6px; font-size: 10.5px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }
        .lpe-specs-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        @media (max-width: 700px) { .lpe-specs-grid { grid-template-columns: repeat(2, 1fr); } }
        .lpe-spec-cell { background: #0d1220; border: 1px solid var(--border2); border-radius: 8px; padding: 8px 10px; display: flex; flex-direction: column; gap: 2px; }
        .lpe-spec-cell span { font-size: 10.5px; color: var(--muted); }
        .lpe-spec-cell b { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--text); }

        .lpe-mono { font-family: 'IBM Plex Mono', monospace; color: var(--text); }
        .lpe-output-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
        .lpe-generate-btn {
          background: var(--amber); color: #16130a; border: none; border-radius: 8px;
          padding: 10px 16px; font-weight: 600; font-size: 13px; cursor: pointer; display: inline-flex; align-items: center; gap: 8px;
          transition: background .15s;
        }
        .lpe-generate-btn:hover { background: #ffbd5c; }
        .lpe-pill {
          display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--teal);
          border: 1px solid #2A3349; border-radius: 20px; padding: 5px 12px; font-family: 'IBM Plex Mono', monospace;
        }
        .lpe-pill-solid { background: var(--amber); color: #16130a; border-color: var(--amber); font-weight: 600; }

        .lpe-mini-pos { display: flex; flex-direction: column; align-items: center; gap: 2px; }
        .lpe-mini-pos span { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--teal); }
        .lpe-mini-trend { display: flex; align-items: center; gap: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--text); }

        .lpe-tl-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
        .lpe-tl { display: flex; flex-direction: column; }
        .lpe-tl-item { display: flex; gap: 14px; opacity: .4; transition: opacity .4s ease; }
        .lpe-tl-item.done { opacity: 1; }
        .lpe-tl-rail { display: flex; flex-direction: column; align-items: center; }
        .lpe-tl-dot {
          width: 28px; height: 28px; border-radius: 50%; border: 2px solid var(--border2); background: var(--panel);
          display: flex; align-items: center; justify-content: center; font-family: 'IBM Plex Mono', monospace;
          font-size: 12px; color: var(--muted); flex-shrink: 0; transition: all .3s ease;
        }
        .lpe-tl-item.done .lpe-tl-dot { border-color: var(--amber); background: #24200f; color: var(--amber); }
        .lpe-tl-item.current .lpe-tl-dot { box-shadow: 0 0 0 4px #f2a93b22; }
        .lpe-tl-line { width: 2px; flex: 1; min-height: 24px; background: var(--border2); margin: 2px 0; }
        .lpe-tl-item.done .lpe-tl-line { background: var(--amber); opacity: .35; }
        .lpe-tl-body { flex: 1; background: #0d1220; border: 1px solid var(--border2); border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; }
        .lpe-tl-body.clickable { cursor: pointer; }
        .lpe-tl-body.clickable:hover { border-color: var(--amber); }
        .lpe-tl-item.done .lpe-tl-body { border-color: #3a3320; }
        .lpe-tl-title { font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
        .lpe-tl-caption { color: var(--muted); font-size: 12px; margin-bottom: 10px; line-height: 1.5; }
        .lpe-tl-waiting { font-style: italic; margin-bottom: 0; }

        .lpe-explain-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        @media (max-width: 700px) { .lpe-explain-grid { grid-template-columns: 1fr; } }
        .lpe-explain-label { font-family: 'Space Grotesk', sans-serif; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--teal); }
        .lpe-explain-grid p { margin: 4px 0 0; font-size: 13.5px; line-height: 1.5; color: #cfd3e0; }
        .lpe-live-note {
          margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border2); display: flex; gap: 8px;
          align-items: flex-start; color: var(--amber); font-size: 12.5px; line-height: 1.5;
        }

        .lpe-footer { margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 11.5px; text-align: center; }
        .lpe-footer b { color: var(--muted); }

        @media (prefers-reduced-motion: reduce) {
          .lpe-token-reveal, .lpe-bar-fill, .lpe-tl-item, .lpe-tl-dot, .lpe-tl-line { transition: none; animation: none; }
        }
      `}</style>

      <div className="lpe-header">
        <div>
          <h1 className="lpe-title"><span className="lpe-logo-dot" />LLM <span>Lens</span></h1>
          <p className="lpe-subtitle">See how a Transformer thinks — live, token by token.</p>
        </div>
        <div className="lpe-input-row">
          <input className="lpe-input" value={draft} onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && commitSentence()} placeholder="Type a sentence…" />
          <button className="lpe-icon-btn" onClick={() => commitSentence()}>Run</button>
          <button className="lpe-icon-btn" onClick={() => commitSentence(DEFAULT_SENTENCE)}><RotateCcw size={14} /></button>
        </div>
        <span className="lpe-badge">Free · No signup</span>
        <div className="lpe-mask-toggle">
          <button className={`lpe-mask-btn ${causal ? 'active' : ''}`} onClick={() => setCausal(true)}><Lock size={12} /> Causal (GPT)</button>
          <button className={`lpe-mask-btn ${!causal ? 'active' : ''}`} onClick={() => setCausal(false)}><Unlock size={12} /> Full (BERT)</button>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="lpe-icon-btn" onClick={() => setActiveIdx((i) => Math.max(0, i - 1))}><SkipBack size={14} /></button>
          <button className="lpe-icon-btn primary" onClick={() => setPlaying((p) => !p)}>
            {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? 'Pause' : 'Play'}
          </button>
          <button className="lpe-icon-btn" onClick={() => setActiveIdx((i) => Math.min(STEPS.length - 1, i + 1))}><SkipForward size={14} /></button>
          <select className="lpe-select" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={2400}>Slow</option>
            <option value={1600}>Normal</option>
            <option value={900}>Fast</option>
          </select>
        </div>
      </div>

      <div className="lpe-presets">
        {PRESETS.map((p) => (
          <button key={p} className="lpe-preset-chip" onClick={() => commitSentence(p)}>{p}</button>
        ))}
      </div>

      <div className="lpe-body">
        <nav className="lpe-rail">
          {STEPS.map((s, i) => (
            <button key={s.id} className={`lpe-rail-item ${i === activeIdx ? 'active' : ''}`} onClick={() => { setActiveIdx(i); setPlaying(false); }}>
              <span className="lpe-rail-num">{s.num}</span>{s.label}
            </button>
          ))}
        </nav>

        <div className="lpe-main">
          <div className="lpe-panel">
            <h2 className="lpe-stage-title">{STEP_INFO[step.id].title}</h2>
            {step.id === 'architecture' && <ArchitectureView />}
            {step.id === 'tokenize' && <TokenizeView tokens={tokens} />}
            {step.id === 'embed' && <EmbedView tokens={tokens} />}
            {step.id === 'position' && <PositionView tokens={tokens} />}
            {step.id === 'attention' && <AttentionView tokens={tokens} selected={selected} setSelected={setSelected} causal={causal} />}
            {step.id === 'layers' && <LayersView tokens={tokens} selected={selected} setSelected={setSelected} causal={causal} />}
            {step.id === 'output' && <OutputView tokens={tokens} sentence={sentence} onGenerate={handleGenerate} />}
            {step.id === 'pipeline' && (
              <PipelineView
                tokens={tokens} sentence={sentence} selected={selected} causal={causal}
                playing={playing} speed={speed} onJump={jumpTo}
                onFinish={() => setPlaying(false)}
              />
            )}
          </div>
          <ExplanationPanel stepId={step.id} extra={extraNote} />
        </div>
      </div>

      <div className="lpe-footer">
        <b>LLM Lens</b> — an educational simulation inspired by "Attention Is All You Need" (Vaswani et al., 2017), for building intuition about Transformer internals. Not a real trained model.
      </div>
    </div>
    </TermContext.Provider>
  );
}